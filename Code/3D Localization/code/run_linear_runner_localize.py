# -*- coding: utf-8 -*-
"""End-to-end 4-view localization for this TPLFM dataset (also runnable as run_lr_localize).

PSF:  zoom15, PSF/file_1..4.tif, 2 um/plane, 256x256
Data: zoom5, Views/{1-4}/suite2p/plane0 (256x256)
FACTOR = (15/5)*(256/256) = 3.0  (parallax)
MOVE_DIVISOR = 3.0  (CA1 original run passed FACTOR into compute_move)
"""
from __future__ import annotations

import csv
import gc
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage

from build_dictionary import build_dictionary
from dataset_config import (
    CROP_HALF_X,
    CROP_HALF_Y,
    DATA_ROOT,
    FACTOR,
    FORCE_RECALIB,
    MOVE_DIVISOR,
    OUT_DIR,
    PIXEL_UM,
    PROJECT,
    PSF_DIR,
    PSF_DZ_UM,
    PSF_ZRANGE_1BASED,
    REFILTER_ONLY,
    T_STRIDE,
    TRACE_T_STRIDE,
    VIEW_CENTERS_OVERRIDE,
    view_plane_dir,
    view_tiff_path,
)
from estimate_depth_other import estimate_depth_other
from merge_dictionary1 import merge_dictionary1
from psf_energy_modification import apply_energy_modification
from psf_fit import psf_fit, stack_views
from psf_seg_TPLFM import CONN4, THRESH_STD, load_psf_volume, psf_seg_TPLFM
from tplfm_utils import compute_move, load_npy

# ---------------------------------------------------------------------------
# Config (dataset_config.py holds paths / FACTOR / MATLAB PSF crop)
# ---------------------------------------------------------------------------
DIST_THR = 40.0
CORR_THR = 0.5
PSF_Z_TRIM = 2  # unused when PSF_ZRANGE_1BASED is set
CROP_HALF_PREFERRED = 90  # legacy square-crop fallback (unused in MATLAB calib)

DEPTH_STEP_UM = 2

# Quality filter after merge.
# ROI shapes can differ across views; a neuron may only be clear in 3/4 views.
# Metrics therefore use the *best k-of-4* views (k = FILTER_MIN_VIEWS).
FILTER_MIN_PEAK_CORR = 0.30  # depth-search peak (sum of up to 3 other-view corrs)
FILTER_MAX_CENTER_SPREAD_PX = 45.0  # max pairwise XY among best k view centers
FILTER_MIN_MEAN_TRACE_CORR = 0.10  # mean pairwise r among best k view traces
FILTER_MIN_VIEWS = 2  # require a usable k-view subset

# If True: only plot traces from existing neuron_dictionary.npy
PLOT_TRACES_ONLY = False
TRACE_PLOT_DIR_NAME = "neuron_traces"


# ---------------------------------------------------------------------------
# suite2p / movies
# ---------------------------------------------------------------------------
def load_suite2p_plane(plane_dir: Path) -> dict[str, Any]:
    iscell = np.asarray(load_npy(plane_dir / "iscell.npy", allow_pickle=False))
    F = np.asarray(load_npy(plane_dir / "F.npy", allow_pickle=False), dtype=np.float64)
    Fneu = np.asarray(load_npy(plane_dir / "Fneu.npy", allow_pickle=False), dtype=np.float64)
    stat = list(load_npy(plane_dir / "stat.npy", allow_pickle=True))
    ops = load_npy(plane_dir / "ops.npy", allow_pickle=True).item()
    if iscell.ndim == 2:
        cell_mask = iscell[:, 0] > 0.5
    else:
        cell_mask = iscell.astype(bool)
    order_1based = np.flatnonzero(cell_mask) + 1
    return {
        "stat": stat,
        "F": F,
        "Fneu": Fneu,
        "ops": ops,
        "iscell": cell_mask,
        "Order_neuron": order_1based,
    }


def _bin_is_blank(bin_path: Path, nframes: int, ly: int, lx: int, dtype) -> bool:
    """True if data.bin appears all-zero (corrupt / empty registration)."""
    raw = np.memmap(str(bin_path), dtype=dtype, mode="r", shape=(nframes, ly, lx))
    # Sample a few frames across the recording
    for i in (0, nframes // 4, nframes // 2, (3 * nframes) // 4, nframes - 1):
        if int(np.max(np.abs(raw[i]))) != 0:
            del raw
            return False
    del raw
    return True


def _load_movie_from_tiff(
    tiff_path: Path,
    ly: int,
    lx: int,
    t_stride: int,
    yoff: np.ndarray | None = None,
    xoff: np.ndarray | None = None,
) -> np.ndarray:
    """Load (Y,X,T) float32 by block-averaging TIFF pages; never hold full T."""
    t_stride = max(int(t_stride), 1)
    yoff_a = None if yoff is None else np.asarray(yoff).ravel()
    xoff_a = None if xoff is None else np.asarray(xoff).ravel()
    with tifffile.TiffFile(str(tiff_path)) as tif:
        n_ok = 0
        for p in tif.pages:
            if not getattr(p, "dataoffsets", ()):
                break
            n_ok += 1
        if n_ok == 0:
            raise ValueError("no readable pages in %s" % tiff_path)
        n_avg = n_ok // t_stride
        if n_avg < 1:
            raise ValueError("TIFF %s has %d pages < t_stride=%d" % (tiff_path, n_ok, t_stride))
        print(
            "  TIFF fallback %s: average %d/%d pages -> %d frames (bin=%d)"
            % (tiff_path.name, n_avg * t_stride, n_ok, n_avg, t_stride)
        )
        movie = np.empty((ly, lx, n_avg), dtype=np.float32)
        for i in range(n_avg):
            acc = np.zeros((ly, lx), dtype=np.float32)
            for j in range(t_stride):
                idx = i * t_stride + j
                frame = np.asarray(tif.pages[idx].asarray(), dtype=np.float32)
                if frame.shape != (ly, lx):
                    raise ValueError(
                        "TIFF page %d shape %s != ops %dx%d" % (idx, frame.shape, ly, lx)
                    )
                if yoff_a is not None and xoff_a is not None and idx < yoff_a.size:
                    dy, dx = int(yoff_a[idx]), int(xoff_a[idx])
                    if dy != 0 or dx != 0:
                        frame = np.roll(frame, shift=(-dy, -dx), axis=(0, 1))
                acc += frame
                del frame
            movie[:, :, i] = acc / float(t_stride)
            del acc
    return movie


def _average_memmap_blocks(
    raw: np.memmap,
    t_stride: int,
) -> np.ndarray:
    """(Y,X,n_avg) float32 from memmap; stream one temporal bin at a time."""
    nframes, ly, lx = int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2])
    t_stride = max(int(t_stride), 1)
    n_avg = nframes // t_stride
    movie = np.empty((ly, lx, n_avg), dtype=np.float32)
    for i in range(n_avg):
        s = i * t_stride
        block = np.asarray(raw[s : s + t_stride], dtype=np.float32)
        movie[:, :, i] = block.mean(axis=0)
        del block
    return movie


def load_views_from_bin(
    data_root: Path = DATA_ROOT,
    t_stride: int = T_STRIDE,
) -> np.ndarray:
    """Load (Y, X, T, 4) float32 by block-averaging data.bin, then free each view.

    If a view's data.bin is blank, falls back to Views/file_{v}/file_{v}.tif
    with rigid registration offsets from ops.
    """
    t_stride = max(int(t_stride), 1)
    ops_list = []
    ly = lx = nframes = None
    for v in range(1, 5):
        plane = view_plane_dir(v, data_root)
        ops = load_npy(plane / "ops.npy", allow_pickle=True).item()
        ly_v, lx_v = int(ops["Ly"]), int(ops["Lx"])
        nframes_v = int(ops["nframes"])
        if ly is None:
            ly, lx, nframes = ly_v, lx_v, nframes_v
        elif ly_v != ly or lx_v != lx:
            raise ValueError("view %d size mismatch: %dx%d vs %dx%d" % (v, ly_v, lx_v, ly, lx))
        elif nframes_v != nframes:
            print("  view%d nframes=%d vs %d; will use min after average" % (v, nframes_v, nframes))
        ops_list.append((v, plane, ops, nframes_v))
    assert ly is not None and lx is not None and nframes is not None
    n_avg = min(nf // t_stride for _v, _p, _o, nf in ops_list)
    out = np.empty((ly, lx, n_avg, 4), dtype=np.float32)

    for v, plane, ops, nframes_v in ops_list:
        bin_path = plane / "data.bin"
        nbytes = bin_path.stat().st_size
        n_pix = ly * lx
        if nbytes == nframes_v * n_pix * 2:
            dtype = np.int16
        elif nbytes == nframes_v * n_pix * 4:
            dtype = np.float32
        else:
            raise ValueError(
                "unexpected data.bin size for view %d: %d bytes (Ly=%d Lx=%d nframes=%d)"
                % (v, nbytes, ly, lx, nframes_v)
            )

        use_tiff = _bin_is_blank(bin_path, nframes_v, ly, lx, dtype)
        if use_tiff:
            tiff_path = view_tiff_path(v, data_root)
            print(
                "  view%d: data.bin is blank - falling back to %s"
                % (v, tiff_path)
            )
            movie = _load_movie_from_tiff(
                tiff_path,
                ly,
                lx,
                t_stride,
                yoff=ops.get("yoff"),
                xoff=ops.get("xoff"),
            )
        else:
            raw = np.memmap(str(bin_path), dtype=dtype, mode="r", shape=(nframes_v, ly, lx))
            movie = _average_memmap_blocks(raw, t_stride)
            del raw
            print(
                "  view%d: averaged %d/%d frames from data.bin -> T=%d"
                % (v, n_avg * t_stride, nframes_v, movie.shape[2])
            )
        if movie.shape[2] < n_avg:
            raise ValueError("view %d averaged T=%d < %d" % (v, movie.shape[2], n_avg))
        out[:, :, :, v - 1] = movie[:, :, :n_avg]
        del movie
        gc.collect()

    print("  packed views %s (%.2f GB)" % (out.shape, out.nbytes / 1e9))
    return out


def apply_integer_move_inplace(
    views: np.ndarray,
    move: np.ndarray,
    s2p: list[dict[str, Any]],
) -> np.ndarray:
    """Align movies + ROI pixels by integer move; return zero move for estimate_depth.

    Matches estimate_depth_other conventions:
      views shifted by (-move[v,1], -move[v,0]) on (Y, X)
      xpix -= move[v,1]; ypix -= move[v,0]
    """
    move = np.asarray(move, dtype=np.float64)
    move_i = np.round(move).astype(int)
    for v in range(views.shape[3]):
        m0, m1 = int(move_i[v, 0]), int(move_i[v, 1])
        if m0 != 0 or m1 != 0:
            views[:, :, :, v] = np.roll(
                views[:, :, :, v], shift=(-m1, -m0), axis=(0, 1)
            )
        for st in s2p[v]["stat"]:
            st["xpix"] = np.asarray(st["xpix"], dtype=np.float64) - m1
            st["ypix"] = np.asarray(st["ypix"], dtype=np.float64) - m0
    return np.zeros_like(move, dtype=np.float64)


# ---------------------------------------------------------------------------
# PSF calibration
# ---------------------------------------------------------------------------
def _auto_center_1based(vol: np.ndarray) -> tuple[int, int]:
    """Largest bright blob centroid on mid Z plane; 1-based (row, col)."""
    iz = vol.shape[2] // 2
    plane = np.asarray(vol[:, :, iz], dtype=np.float64)
    plane = plane - plane.mean()
    stdd = float(plane.std(ddof=1))
    mask = plane >= (THRESH_STD * stdd if stdd > 0 else 0.0)
    labeled, n_lab = ndimage.label(mask, structure=CONN4)
    if n_lab == 0:
        return (vol.shape[0] // 2 + 1, vol.shape[1] // 2 + 1)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    ind = int(np.argmax(counts))
    ys, xs = np.nonzero(labeled == ind)
    return (int(round(ys.mean())) + 1, int(round(xs.mean())) + 1)


def _safe_crop_half(n_y: int, n_x: int, cy0: int, cx0: int, preferred: int) -> int:
    """Max half-size that fits around 0-based center, capped at preferred."""
    cy, cx = cy0, cx0
    max_h = min(cy, n_y - 1 - cy, cx, n_x - 1 - cx, preferred)
    return max(int(max_h), 8)


def calibrate_psf(
    data_root: Path = DATA_ROOT,
    out_dir: Path = OUT_DIR,
    *,
    psf_dir: Path = PSF_DIR,
    z_trim: int = PSF_Z_TRIM,
    crop_half_y: int = CROP_HALF_Y,
    crop_half_x: int = CROP_HALF_X,
    dz_um: float = PSF_DZ_UM,
    centers_override: list[tuple[int, int]] | None = VIEW_CENTERS_OVERRIDE,
    zrange_1based: range | None = PSF_ZRANGE_1BASED,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_list: list[np.ndarray] = []
    centers_1based: list[tuple[int, int]] = []
    volumes = []
    for v in range(1, 5):
        path = psf_dir / ("file_%d.tif" % v)
        if not path.is_file():
            path = PROJECT / ("file_%d.tif" % v)
        print("Loading PSF %s ..." % path)
        vol = load_psf_volume(path)
        volumes.append(vol)
        print("  shape (Y,X,Z)=%s" % (vol.shape,))

    nz = int(volumes[0].shape[2])
    if zrange_1based is not None:
        zrange = zrange_1based
    else:
        z_start = z_trim + 1
        z_end = nz - z_trim
        if z_end < z_start:
            raise ValueError("PSF Z trim too aggressive: start=%d end=%d" % (z_start, z_end))
        zrange = range(z_start, z_end + 1)
    z_lo, z_hi = int(min(zrange)), int(max(zrange))
    print("PSF z (1-based) = %d:%d  (%d planes)  crop %dx%d" % (
        z_lo, z_hi, z_hi - z_lo + 1, 2 * crop_half_y + 1, 2 * crop_half_x + 1
    ))

    for v, vol in enumerate(volumes, start=1):
        if centers_override is not None:
            center = tuple(int(x) for x in centers_override[v - 1])
        else:
            center = _auto_center_1based(vol)
        centers_1based.append(center)
        print(
            "  view%d center(1-based row,col)=%s crop_half_y=%d crop_half_x=%d"
            % (v, center, crop_half_y, crop_half_x)
        )
        seg = psf_seg_TPLFM(
            vol,
            center,
            zrange,
            one_based=True,
            crop_half_y=crop_half_y,
            crop_half_x=crop_half_x,
        )
        seg_list.append(seg)
        tifffile.imwrite(
            str(out_dir / ("psf%d_seg.tif" % v)),
            np.transpose(seg, (2, 0, 1)),
        )

    balanced, xpsf = apply_energy_modification(seg_list)
    for v, vol in enumerate(balanced, start=1):
        tifffile.imwrite(
            str(out_dir / ("psf%d_energy.tif" % v)),
            np.transpose(vol, (2, 0, 1)),
        )
    np.save(str(out_dir / "xpsf.npy"), xpsf)

    n_z = balanced[0].shape[2]
    z_um = (np.arange(n_z, dtype=np.float64) - (n_z - 1) / 2.0) * float(dz_um)
    fit_line, coef_xz, coef_yz, coef_xy, x0, y0 = psf_fit(
        stack_views(balanced), z_um=z_um
    )
    coef_path = out_dir / "psf_fit_coef.npz"
    np.savez(
        str(coef_path),
        fit_line=fit_line,
        coef_psf_xz=coef_xz,
        coef_psf_yz=coef_yz,
        coef_psf_xy=coef_xy,
        x0=x0,
        y0=y0,
        xpsf=xpsf,
        z_um=z_um,
        centers_1based=np.asarray(centers_1based, dtype=np.float64),
        crop_half_y=int(crop_half_y),
        crop_half_x=int(crop_half_x),
        dz_um=float(dz_um),
        factor=float(FACTOR),
        move_divisor=float(MOVE_DIVISOR),
    )
    print("coef_psf_xz slopes:", coef_xz[0])
    print("coef_psf_yz slopes:", coef_yz[0])
    print("x0=%.2f y0=%.2f  z_um=[%.1f, %.1f]" % (x0, y0, z_um[0], z_um[-1]))
    print("saved", coef_path)

    _plot_psf_trajectories(fit_line, z_um, coef_xz, coef_yz, out_dir)
    return {
        "coef_xz": coef_xz,
        "coef_yz": coef_yz,
        "coef_xy": coef_xy,
        "x0": x0,
        "y0": y0,
        "xpsf": xpsf,
        "z_um": z_um,
        "fit_line": fit_line,
        "centers_1based": np.asarray(centers_1based, dtype=np.float64),
        "crop_half_y": int(crop_half_y),
        "crop_half_x": int(crop_half_x),
    }


def _plot_psf_trajectories(
    fit_line: np.ndarray,
    z_um: np.ndarray,
    coef_xz: np.ndarray,
    coef_yz: np.ndarray,
    out_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing — skip psf_fit_trajectories.png")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for v in range(4):
        ax = axes.ravel()[v]
        ax.plot(z_um, fit_line[0, :, v], "o-", ms=3, label="x (col)")
        ax.plot(z_um, fit_line[1, :, v], "s-", ms=3, label="y (row)")
        ax.plot(z_um, np.polyval(coef_xz[:, v], z_um), "C0--", lw=1)
        ax.plot(z_um, np.polyval(coef_yz[:, v], z_um), "C1--", lw=1)
        ax.set_title("view %d" % (v + 1))
        ax.set_xlabel("z (um)")
        ax.set_ylabel("centroid (px, 1-based)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("PSF centroid trajectories (TPLFM LinearRunner)")
    fig.tight_layout()
    path = out_dir / "psf_fit_trajectories.png"
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Localization
# ---------------------------------------------------------------------------
def repair_traces_from_F(
    dicts: list[list[dict[str, Any]]],
    s2p: list[dict[str, Any]],
    *,
    t_stride: int,
    n_t: int,
) -> None:
    """Replace near-zero movie traces with suite2p F (same temporal stride)."""
    for d, sp in zip(dicts, s2p):
        F = np.asarray(sp["F"], dtype=np.float64)
        order = np.asarray(sp["Order_neuron"]).astype(int).ravel()
        for j, entry in enumerate(d):
            tr = np.asarray(entry["trace"], dtype=np.float64)
            if tr.size and float(np.std(tr)) > 1e-3:
                continue
            oi = int(order[j]) - 1
            if oi < 0 or oi >= F.shape[0]:
                continue
            raw_f = F[oi]
            n_avg = int(raw_f.size // t_stride)
            if n_avg < 1:
                continue
            ftr = raw_f[: n_avg * t_stride].reshape(n_avg, t_stride).mean(axis=1)[:n_t]
            if ftr.size < n_t:
                pad = np.zeros(n_t, dtype=np.float64)
                pad[: ftr.size] = ftr
                ftr = pad
            entry["trace"] = ftr
            print("  repaired trace view%d neuron%d from F.npy" % (entry.get("primary_view", -1), j))


def localize_all_views(
    views: np.ndarray,
    s2p: list[dict[str, Any]],
    coef_xz: np.ndarray,
    coef_yz: np.ndarray,
    move: np.ndarray,
    *,
    factor: float,
    x0: float,
    y0: float,
    depth_min: int,
    depth_max: int,
    depth_step: int = DEPTH_STEP_UM,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    dicts = []
    peak_corrs: list[np.ndarray] = []
    for view in range(1, 5):
        print(
            "estimate_depth_other primary view=%d (iscell n=%d) ..."
            % (view, s2p[view - 1]["Order_neuron"].size)
        )
        sp = s2p[view - 1]
        viewi = views[:, :, :, view - 1]
        (
            neuron_centeri,
            cori,
            _depth_neuron,
            rawtrace,
            _masks,
            pixels,
            centers_allview,
        ) = estimate_depth_other(
            sp["stat"],
            sp["F"],
            sp["Fneu"],
            sp["Order_neuron"],
            coef_xz,
            coef_yz,
            viewi,
            views,
            view,
            move,
            factor=factor,
            pix_one_based=False,
            x0=x0,
            y0=y0,
            depth_min=depth_min,
            depth_max=depth_max,
            depth_step=depth_step,
        )
        d = build_dictionary(neuron_centeri, rawtrace, centers_allview, pixels, cori)
        peak = np.nanmax(cori, axis=0) if cori.size else np.zeros(0)
        peak_corrs.append(peak)
        for i, entry in enumerate(d):
            entry["peak_corr"] = float(peak[i]) if i < peak.size else float("nan")
            entry["primary_view"] = view
        print("  view%d neurons=%d" % (view, len(d)))
        dicts.append(d)
    return dicts, peak_corrs


def _view_centers_xy(entry: dict[str, Any]) -> list[tuple[float, float] | None]:
    cav = np.asarray(entry["center_allview"], dtype=np.float64)
    if cav.shape != (2, 4):
        cav = cav.reshape(2, 4)
    out: list[tuple[float, float] | None] = []
    for v in range(4):
        x, y = float(cav[0, v]), float(cav[1, v])
        if np.isfinite(x) and np.isfinite(y) and (abs(x) > 1e-6 or abs(y) > 1e-6):
            out.append((x, y))
        else:
            out.append(None)
    return out


def _max_pairwise_dist(pts: list[tuple[float, float]]) -> float:
    dmax = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
            if d > dmax:
                dmax = d
    return dmax


def _best_k_center_spread(
    entry: dict[str, Any],
    k: int = FILTER_MIN_VIEWS,
) -> tuple[float, tuple[int, ...]]:
    """Smallest max-pairwise XY spread among any k finite view centers."""
    centers = _view_centers_xy(entry)
    idxs = [i for i, p in enumerate(centers) if p is not None]
    if len(idxs) < k:
        return float("inf"), tuple()
    from itertools import combinations

    best_spread = float("inf")
    best_set: tuple[int, ...] = tuple()
    for comb in combinations(idxs, k):
        pts = [centers[i] for i in comb]  # type: ignore[misc]
        sp = _max_pairwise_dist(pts)  # type: ignore[arg-type]
        if sp < best_spread:
            best_spread = sp
            best_set = comb
    return best_spread, best_set


def _best_k_trace_corr(
    traces: np.ndarray,
    k: int = FILTER_MIN_VIEWS,
) -> tuple[float, tuple[int, ...]]:
    """Highest mean pairwise Pearson r among any k views (ignore one bad view)."""
    from itertools import combinations

    from tplfm_utils import corrcoef_scalar

    usable = [i for i in range(4) if float(np.std(traces[i])) > 1e-6]
    if len(usable) < k:
        return float("nan"), tuple()
    best_mean = -np.inf
    best_set: tuple[int, ...] = tuple()
    for comb in combinations(usable, k):
        rs = []
        for a, b in combinations(comb, 2):
            r = corrcoef_scalar(traces[a], traces[b])
            if np.isfinite(r):
                rs.append(float(r))
        if not rs:
            continue
        m = float(np.mean(rs))
        if m > best_mean:
            best_mean = m
            best_set = comb
    if best_set == tuple():
        return float("nan"), tuple()
    return float(best_mean), best_set


def filter_merged_neurons(
    merged: list[dict[str, Any]],
    traces_n4t: np.ndarray | None = None,
    *,
    min_peak_corr: float = FILTER_MIN_PEAK_CORR,
    max_center_spread_px: float = FILTER_MAX_CENTER_SPREAD_PX,
    min_mean_trace_corr: float = FILTER_MIN_MEAN_TRACE_CORR,
    min_views: int = FILTER_MIN_VIEWS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep neurons with a usable best-k-of-4 subset (default k=3).

    Does NOT require all 4 views or identical ROI shapes. One outlier view is ignored
    when scoring center spread and footprint-trace correlation.
    """
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reject_info: list[dict[str, Any]] = []

    for i, entry in enumerate(merged):
        peak = float(entry.get("peak_corr", np.nan))
        if not np.isfinite(peak) and "cori_allneuron_allz" in entry:
            curve = np.asarray(entry["cori_allneuron_allz"], dtype=np.float64)
            if curve.size:
                peak = float(np.nanmax(curve))

        spread, spread_views = _best_k_center_spread(entry, k=min_views)
        mean_r, corr_views = float("nan"), tuple()
        if traces_n4t is not None:
            mean_r, corr_views = _best_k_trace_corr(traces_n4t[i], k=min_views)

        n_finite = sum(1 for p in _view_centers_xy(entry) if p is not None)
        reasons = []
        if n_finite < min_views:
            reasons.append("n_views=%d<%d" % (n_finite, min_views))
        if not np.isfinite(peak) or peak < min_peak_corr:
            reasons.append("peak_corr=%.3f<%.3f" % (peak, min_peak_corr))
        if not np.isfinite(spread) or spread > max_center_spread_px:
            reasons.append(
                "best%d_spread=%.1f>%.1f" % (min_views, spread, max_center_spread_px)
            )
        if traces_n4t is not None:
            if not np.isfinite(mean_r) or mean_r < min_mean_trace_corr:
                reasons.append(
                    "best%d_mean_r=%.3f<%.3f" % (min_views, mean_r, min_mean_trace_corr)
                )

        meta = {
            "orig_idx": i,
            "peak_corr": peak,
            "center_spread_px": spread,
            "mean_trace_corr": mean_r,
            "min_trace_corr": mean_r,  # kept for CSV compat; now = best-k mean
            "best_center_views": ",".join(str(v + 1) for v in spread_views),
            "best_corr_views": ",".join(str(v + 1) for v in corr_views),
            "reasons": "; ".join(reasons) if reasons else "",
        }
        entry = dict(entry)
        entry["peak_corr"] = peak
        entry["center_spread_px"] = spread
        entry["mean_trace_corr"] = mean_r
        entry["min_trace_corr"] = mean_r
        entry["best_center_views"] = meta["best_center_views"]
        entry["best_corr_views"] = meta["best_corr_views"]
        entry["orig_idx"] = i

        if reasons:
            rejected.append(entry)
            reject_info.append(meta)
        else:
            kept.append(entry)

    print(
        "filter (best-%d-of-4): keep %d / reject %d  "
        "(peak>=%.2f, spread<=%.1fpx, mean_r>=%.2f)"
        % (
            min_views,
            len(kept),
            len(rejected),
            min_peak_corr,
            max_center_spread_px,
            min_mean_trace_corr,
        )
    )
    if reject_info:
        for row in reject_info[:8]:
            print("  reject orig#%d: %s" % (row["orig_idx"], row["reasons"]))
        if len(reject_info) > 8:
            print("  ... (%d more)" % (len(reject_info) - 8))
    return kept, rejected, reject_info


def _n_views_matched(entry: dict[str, Any]) -> int:
    """Count views with finite non-zero projected center (heuristic)."""
    cav = np.asarray(entry.get("center_allview", np.zeros((2, 4))), dtype=np.float64)
    if cav.shape != (2, 4):
        return 1
    # Non-zero in either row/col for a view counts as present
    return int(np.sum(np.any(np.abs(cav) > 1e-6, axis=0)))


def export_results(
    merged: list[dict[str, Any]],
    out_dir: Path,
    *,
    pixel_um: float = PIXEL_UM,
    dz_um: float = PSF_DZ_UM,
) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "neuron_dictionary.npy"), np.array(merged, dtype=object))

    rows = []
    coords = np.zeros((len(merged), 3), dtype=np.float64)
    for i, e in enumerate(merged):
        c = np.asarray(e["center"], dtype=np.float64).ravel()
        col, row, z_um = float(c[0]), float(c[1]), float(c[2])
        coords[i] = (col, row, z_um)
        peak = float(e.get("peak_corr", np.nan))
        if not np.isfinite(peak) and "cori_allneuron_allz" in e:
            curve = np.asarray(e["cori_allneuron_allz"], dtype=np.float64)
            if curve.size:
                peak = float(np.nanmax(curve))
        rows.append(
            {
                "idx": i,
                "orig_idx": int(e.get("orig_idx", i)),
                "col": col,
                "row": row,
                "z_um": z_um,
                "z_plane_equiv": z_um / float(dz_um),
                "x_um": col * float(pixel_um),
                "y_um": row * float(pixel_um),
                "n_views_matched": _n_views_matched(e),
                "peak_corr": peak,
                "center_spread_px": float(e.get("center_spread_px", np.nan)),
                "mean_trace_corr": float(e.get("mean_trace_corr", np.nan)),
                "min_trace_corr": float(e.get("min_trace_corr", np.nan)),
                "primary_view": int(e.get("primary_view", -1)),
            }
        )

    csv_path = out_dir / "localization.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "idx",
            "orig_idx",
            "col",
            "row",
            "z_um",
            "z_plane_equiv",
            "x_um",
            "y_um",
            "n_views_matched",
            "peak_corr",
            "center_spread_px",
            "mean_trace_corr",
            "min_trace_corr",
            "primary_view",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)

    if coords.size == 0:
        return coords

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing — skip localization plots")
        return coords

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=coords[:, 2], cmap="viridis", s=28)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("col (px)")
    ax.set_ylabel("row (px)")
    ax.set_title("TPLFM LinearRunner XY (color = z um)")
    fig.colorbar(sc, ax=ax, label="z (um)")
    fig.tight_layout()
    p = out_dir / "localization_xy.png"
    fig.savefig(str(p), dpi=120)
    plt.close(fig)
    print("wrote", p)

    fig = plt.figure(figsize=(7, 6))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=coords[:, 2], cmap="viridis", s=28)
    ax3.set_xlabel("col (px)")
    ax3.set_ylabel("row (px)")
    ax3.set_zlabel("z (um)")
    ax3.set_title("TPLFM LinearRunner 3D localization")
    fig.tight_layout()
    p = out_dir / "localization_3d.png"
    fig.savefig(str(p), dpi=120)
    plt.close(fig)
    print("wrote", p)

    fig, ax = plt.subplots(figsize=(6, 4))
    z_finite = coords[:, 2][np.isfinite(coords[:, 2])]
    if z_finite.size:
        ax.hist(z_finite, bins=min(40, max(8, z_finite.size // 2)), color="C0", edgecolor="k")
    ax.set_xlabel("z (um)")
    ax.set_ylabel("count")
    ax.set_title("Depth histogram (n=%d finite)" % z_finite.size)
    fig.tight_layout()
    p = out_dir / "depth_hist.png"
    fig.savefig(str(p), dpi=120)
    plt.close(fig)
    print("wrote", p)

    peak_vals = np.asarray([r["peak_corr"] for r in rows], dtype=np.float64)
    peak_finite = peak_vals[np.isfinite(peak_vals)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if peak_finite.size:
        ax.hist(
            peak_finite,
            bins=min(40, max(10, peak_finite.size // 5)),
            color="C0",
            edgecolor="k",
            alpha=0.85,
        )
        ax.axvline(
            float(np.median(peak_finite)),
            color="C3",
            ls="--",
            lw=1.5,
            label="median=%.3f" % float(np.median(peak_finite)),
        )
        ax.axvline(
            float(peak_finite.mean()),
            color="C1",
            ls=":",
            lw=1.5,
            label="mean=%.3f" % float(peak_finite.mean()),
        )
        ax.legend()
    ax.set_xlabel("peak_corr")
    ax.set_ylabel("count")
    ax.set_title("peak_corr distribution (n=%d)" % peak_finite.size)
    fig.tight_layout()
    p = out_dir / "peak_corr_hist.png"
    fig.savefig(str(p), dpi=140)
    plt.close(fig)
    print("wrote", p)

    return coords


def open_view_memmaps(
    data_root: Path = DATA_ROOT,
) -> tuple[list[np.memmap], int, int, int, list[tuple[int, int]]]:
    """Open 4 suite2p data.bin as memmaps. Returns memmaps, Ly, Lx, nframes, move_pairs unused.

    Raises if a view's data.bin is blank (use a real registered binary).
    """
    mems: list[np.memmap] = []
    ly = lx = nframes = None
    for v in range(1, 5):
        plane = view_plane_dir(v, data_root)
        ops = load_npy(plane / "ops.npy", allow_pickle=True).item()
        ly_v, lx_v = int(ops["Ly"]), int(ops["Lx"])
        nf = int(ops["nframes"])
        if ly is None:
            ly, lx, nframes = ly_v, lx_v, nf
        elif (ly_v, lx_v, nf) != (ly, lx, nframes):
            raise ValueError(
                "view %d shape/frames mismatch: %dx%d x%d vs %dx%d x%d"
                % (v, ly_v, lx_v, nf, ly, lx, nframes)
            )
        bin_path = plane / "data.bin"
        nbytes = bin_path.stat().st_size
        n_pix = ly * lx
        if nbytes == nf * n_pix * 2:
            dtype = np.int16
        elif nbytes == nf * n_pix * 4:
            dtype = np.float32
        else:
            raise ValueError("bad data.bin size view %d" % v)
        if _bin_is_blank(bin_path, nf, ly, lx, dtype):
            raise RuntimeError(
                "view %d data.bin is blank; restore registered data.bin before plotting"
                % v
            )
        mems.append(np.memmap(str(bin_path), dtype=dtype, mode="r", shape=(nf, ly, lx)))
        print("  view%d memmap %s frames=%d" % (v, bin_path, nf))
    return mems, int(ly), int(lx), int(nframes)


def mean_images_from_memmaps(
    mems: list[np.memmap],
    move: np.ndarray,
    *,
    sample_stride: int = 40,
) -> np.ndarray:
    """(Y,X,4) mean images in the same rolled coordinates as localization."""
    nf, ly, lx = mems[0].shape
    move_i = np.round(np.asarray(move, dtype=np.float64)).astype(int)
    out = np.zeros((ly, lx, 4), dtype=np.float32)
    for v, raw in enumerate(mems):
        acc = np.zeros((ly, lx), dtype=np.float64)
        n = 0
        for i in range(0, nf, sample_stride):
            acc += np.asarray(raw[i], dtype=np.float64)
            n += 1
        img = (acc / max(n, 1)).astype(np.float32)
        m0, m1 = int(move_i[v, 0]), int(move_i[v, 1])
        if m0 or m1:
            img = np.roll(img, shift=(-m1, -m0), axis=(0, 1))
        out[:, :, v] = img
    return out


def extract_all_traces_memmap(
    merged: list[dict[str, Any]],
    mems: list[np.memmap],
    move: np.ndarray,
    *,
    t_stride: int = TRACE_T_STRIDE,
    frame_chunk: int = 100,
) -> np.ndarray:
    """Extract traces for all neurons: (N, 4, T), one pass per view over data.bin."""
    n = len(merged)
    nf, ly, lx = mems[0].shape
    frame_idx = np.arange(0, nf, t_stride)
    n_t = int(frame_idx.size)
    move_i = np.round(np.asarray(move, dtype=np.float64)).astype(int)
    out = np.zeros((n, 4, n_t), dtype=np.float64)

    # Precompute rolled->original ROI pixels per neuron/view
    pix: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for entry in merged:
        cav = np.asarray(entry["center_allview"], dtype=np.float64)
        if cav.shape != (2, 4):
            cav = cav.reshape(2, 4)
        d1 = np.asarray(entry["pixels1"], dtype=np.float64).ravel()
        d2 = np.asarray(entry["pixels2"], dtype=np.float64).ravel()
        row = []
        for v in range(4):
            cx, cy = float(cav[0, v]), float(cav[1, v])
            if not (np.isfinite(cx) and np.isfinite(cy)) or d1.size == 0:
                row.append((np.zeros(0, dtype=int), np.zeros(0, dtype=int)))
                continue
            px = np.rint(d1 + cx).astype(int)
            py = np.rint(d2 + cy).astype(int)
            m0, m1 = int(move_i[v, 0]), int(move_i[v, 1])
            ox = np.clip(px + m1, 0, lx - 1)
            oy = np.clip(py + m0, 0, ly - 1)
            row.append((oy, ox))
        pix.append(row)

    for v in range(4):
        raw = mems[v]
        print("  extracting view%d traces (%d neurons, T=%d) ..." % (v + 1, n, n_t))
        for s in range(0, n_t, frame_chunk):
            e = min(s + frame_chunk, n_t)
            fi = frame_idx[s:e]
            block = np.asarray(raw[fi], dtype=np.float32)  # (chunk, Y, X)
            for ni in range(n):
                oy, ox = pix[ni][v]
                if oy.size == 0:
                    continue
                out[ni, v, s:e] = block[:, oy, ox].mean(axis=1)
    return out


def extract_four_view_traces_memmap(
    entry: dict[str, Any],
    mems: list[np.memmap],
    move: np.ndarray,
    *,
    t_stride: int = TRACE_T_STRIDE,
) -> np.ndarray:
    """Extract (4, T) for a single neuron (wrapper around batch extractor)."""
    return extract_all_traces_memmap([entry], mems, move, t_stride=t_stride)[0]


def extract_four_view_traces(
    entry: dict[str, Any],
    views: np.ndarray,
) -> np.ndarray:
    """Extract (4, T) mean traces at center_allview + footprint deltas from (Y,X,T,4)."""
    h, w, n_t, n_views = views.shape
    cav = np.asarray(entry["center_allview"], dtype=np.float64)
    if cav.shape != (2, 4):
        cav = cav.reshape(2, 4)
    d1 = np.asarray(entry["pixels1"], dtype=np.float64).ravel()
    d2 = np.asarray(entry["pixels2"], dtype=np.float64).ravel()
    traces = np.zeros((n_views, n_t), dtype=np.float64)
    for v in range(n_views):
        cx, cy = float(cav[0, v]), float(cav[1, v])
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        px = np.clip(np.rint(d1 + cx).astype(int), 0, w - 1)
        py = np.clip(np.rint(d2 + cy).astype(int), 0, h - 1)
        if px.size == 0:
            continue
        traces[v] = views[py, px, :, v].mean(axis=0)
    return traces


def footprint_coords(
    entry: dict[str, Any],
    view_idx: int,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (cols, rows, cx, cy) for one view footprint in image coordinates."""
    cav = np.asarray(entry["center_allview"], dtype=np.float64)
    if cav.shape != (2, 4):
        cav = cav.reshape(2, 4)
    d1 = np.asarray(entry["pixels1"], dtype=np.float64).ravel()
    d2 = np.asarray(entry["pixels2"], dtype=np.float64).ravel()
    cx, cy = float(cav[0, view_idx]), float(cav[1, view_idx])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), cx, cy
    cols = np.clip(np.rint(d1 + cx).astype(int), 0, w - 1)
    rows = np.clip(np.rint(d2 + cy).astype(int), 0, h - 1)
    return cols, rows, cx, cy


def _zscore(tr: np.ndarray) -> np.ndarray:
    tr = np.asarray(tr, dtype=np.float64)
    s = float(np.std(tr))
    if s < 1e-12:
        return np.zeros_like(tr)
    return (tr - float(np.mean(tr))) / s


def save_neuron_trace_plots(
    merged: list[dict[str, Any]],
    out_dir: Path,
    *,
    views: np.ndarray | None = None,
    mems: list | None = None,
    move: np.ndarray | None = None,
    mean_imgs: np.ndarray | None = None,
    t_stride: int = TRACE_T_STRIDE,
    subdir: str = TRACE_PLOT_DIR_NAME,
) -> Path:
    """Save one PNG per neuron: footprints + traces.

    Prefer mems+move for full-length data.bin traces (low RAM).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.spatial import ConvexHull

    from tplfm_utils import corrcoef_scalar

    plot_dir = out_dir / subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    if mems is not None:
        if move is None:
            raise ValueError("move is required with mems")
        h, w = int(mems[0].shape[1]), int(mems[0].shape[2])
        if mean_imgs is None:
            mean_imgs = mean_images_from_memmaps(mems, move)
        bg = np.max(mean_imgs, axis=2)
        print(
            "trace plots: n=%d  T~%d (TRACE stride=%d, nframes=%d)"
            % (
                len(merged),
                len(range(0, mems[0].shape[0], t_stride)),
                t_stride,
                mems[0].shape[0],
            )
        )
        all_traces = extract_all_traces_memmap(
            merged, mems, move, t_stride=t_stride
        )
    elif views is not None:
        h, w, _, _ = views.shape
        mean_imgs = views.mean(axis=2)
        bg = np.max(mean_imgs, axis=2)
        print("trace plots: n=%d from views array T=%d" % (len(merged), views.shape[2]))
        all_traces = np.stack(
            [extract_four_view_traces(e, views) for e in merged], axis=0
        )
    else:
        raise ValueError("provide views or mems")

    for i, entry in enumerate(merged):
        traces = all_traces[i]
        n_t = traces.shape[1]
        t = np.arange(n_t)
        c = np.asarray(entry["center"], dtype=np.float64).ravel()
        peak = float(entry.get("peak_corr", np.nan))
        if not np.isfinite(peak) and "cori_allneuron_allz" in entry:
            curve = np.asarray(entry["cori_allneuron_allz"], dtype=np.float64)
            if curve.size:
                peak = float(np.nanmax(curve))
        primary = int(entry.get("primary_view", -1))
        ref_v = primary - 1 if 1 <= primary <= 4 else 0
        corrs = [corrcoef_scalar(traces[ref_v], traces[v]) for v in range(4)]
        fps = [footprint_coords(entry, v, h, w) for v in range(4)]

        fig = plt.figure(figsize=(12, 13))
        gs = fig.add_gridspec(
            6,
            2,
            height_ratios=[2.8, 1.15, 1, 1, 1, 1],
            width_ratios=[1.15, 1],
            hspace=0.38,
            wspace=0.25,
        )
        ax_fp = fig.add_subplot(gs[0, 0])
        ax_fp2 = fig.add_subplot(gs[0, 1])
        ax0 = fig.add_subplot(gs[1, :])
        axes_tr = [fig.add_subplot(gs[2 + v, :], sharex=ax0) for v in range(4)]

        if any(fp[0].size for fp in fps):
            all_cols = np.concatenate([fp[0] for fp in fps if fp[0].size])
            all_rows = np.concatenate([fp[1] for fp in fps if fp[1].size])
            pad = 50
            x0 = int(max(0, all_cols.min() - pad))
            x1 = int(min(w, all_cols.max() + pad + 1))
            y0 = int(max(0, all_rows.min() - pad))
            y1 = int(min(h, all_rows.max() + pad + 1))
        else:
            x0, x1, y0, y1 = 0, w, 0, h

        crop = bg[y0:y1, x0:x1]
        ax_fp.imshow(
            crop,
            cmap="gray",
            origin="upper",
            extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5),
            vmin=np.percentile(crop, 2),
            vmax=np.percentile(crop, 98),
        )
        legend_handles = []
        for v, (cols, rows, cx, cy) in enumerate(fps):
            if cols.size == 0:
                continue
            ax_fp.scatter(
                cols, rows, s=6, c=colors[v], alpha=0.18, linewidths=0, zorder=2 + v
            )
            pts = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])
            if pts.shape[0] >= 3:
                try:
                    hull = ConvexHull(pts)
                    hull_pts = np.vstack([pts[hull.vertices], pts[hull.vertices][0]])
                    ax_fp.plot(
                        hull_pts[:, 0],
                        hull_pts[:, 1],
                        color=colors[v],
                        lw=2.0,
                        zorder=10 + v,
                    )
                except Exception:
                    pass
            if np.isfinite(cx) and np.isfinite(cy):
                ax_fp.plot(
                    cx, cy, marker="x", color=colors[v], ms=10, mew=2.2, zorder=20 + v
                )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[v],
                    lw=2.0,
                    marker="x",
                    markersize=7,
                    label="v%d (n=%d, r=%.2f)" % (v + 1, cols.size, corrs[v]),
                )
            )
        ax_fp.set_xlim(x0, x1)
        ax_fp.set_ylim(y1, y0)
        ax_fp.set_aspect("equal")
        ax_fp.set_xlabel("col (px)")
        ax_fp.set_ylabel("row (px)")
        ax_fp.set_title("merged footprints (fill+hull+center)")
        if legend_handles:
            ax_fp.legend(handles=legend_handles, loc="upper right", fontsize=8)

        ax_fp2.set_axis_off()
        inner = gs[0, 1].subgridspec(2, 2, wspace=0.15, hspace=0.25)
        for v, (cols, rows, cx, cy) in enumerate(fps):
            ax = fig.add_subplot(inner[v // 2, v % 2])
            ax.imshow(
                crop,
                cmap="gray",
                origin="upper",
                extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5),
                vmin=np.percentile(crop, 2),
                vmax=np.percentile(crop, 98),
            )
            if cols.size:
                ax.scatter(cols, rows, s=5, c=colors[v], alpha=0.55, linewidths=0)
            if np.isfinite(cx) and np.isfinite(cy):
                ax.plot(cx, cy, "x", color=colors[v], ms=8, mew=2)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_aspect("equal")
            ax.set_title("v%d" % (v + 1), color=colors[v], fontsize=10)
            ax.tick_params(labelsize=7)

        fig.suptitle(
            "neuron %03d | z=%.1f um | peak_corr=%.3f | primary=v%d | T=%d"
            % (i, float(c[2]) if c.size > 2 else float("nan"), peak, primary, n_t),
            fontsize=12,
            y=0.995,
        )

        for v in range(4):
            ax0.plot(
                t,
                _zscore(traces[v]),
                color=colors[v],
                lw=0.55,
                label="v%d r=%.2f" % (v + 1, corrs[v]),
            )
        ax0.set_ylabel("z-score")
        ax0.legend(loc="upper right", fontsize=8, ncol=4)
        ax0.grid(True, alpha=0.3)
        ax0.set_title("4-view traces (z-scored, %d frames)" % n_t)

        for v in range(4):
            ax = axes_tr[v]
            ax.plot(t, traces[v], color=colors[v], lw=0.45)
            ax.set_ylabel("v%d" % (v + 1))
            ax.grid(True, alpha=0.3)
            ax.text(
                0.01,
                0.92,
                "std=%.1f  corr(ref)=%.3f" % (float(np.std(traces[v])), corrs[v]),
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )
        axes_tr[-1].set_xlabel(
            "frame" if t_stride == 1 else "frame (stride=%d)" % t_stride
        )

        fig.subplots_adjust(hspace=0.4, top=0.96, bottom=0.05, left=0.08, right=0.98)
        path_out = plot_dir / ("neuron_%03d_traces.png" % i)
        try:
            fig.savefig(str(path_out), dpi=100)
        except OSError as exc:
            print("  skip trace plot %s (%s)" % (path_out.name, exc))
        plt.close(fig)
        if (i + 1) % 25 == 0 or i == 0 or i + 1 == len(merged):
            print(
                "  wrote traces %d/%d -> %s (T=%d)"
                % (i + 1, len(merged), path_out.name, n_t)
            )

    print("wrote %d trace figures in %s" % (len(merged), plot_dir))
    return plot_dir


def refilter_from_saved_all() -> None:
    """Re-apply quality filter on neuron_dictionary_all.npy and replot kept only."""
    all_path = OUT_DIR / "neuron_dictionary_all.npy"
    if not all_path.is_file():
        raise FileNotFoundError(all_path)
    merged = list(np.load(str(all_path), allow_pickle=True))
    print("loaded %d neurons from %s" % (len(merged), all_path))

    coef_path = OUT_DIR / "psf_fit_coef.npz"
    data = np.load(str(coef_path))
    centers = np.asarray(data["centers_1based"], dtype=np.float64)
    move = compute_move(centers, factor=MOVE_DIVISOR)

    mems, ly, lx, nf = open_view_memmaps(DATA_ROOT)
    print("FOV %dx%d, nframes=%d" % (ly, lx, nf))
    traces_for_filter = extract_all_traces_memmap(
        merged, mems, move, t_stride=max(T_STRIDE, 1)
    )
    kept, rejected, reject_info = filter_merged_neurons(merged, traces_for_filter)

    rej_csv = OUT_DIR / "localization_rejected.csv"
    with open(rej_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "orig_idx",
            "peak_corr",
            "center_spread_px",
            "mean_trace_corr",
            "min_trace_corr",
            "best_center_views",
            "best_corr_views",
            "reasons",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(reject_info)
    print("wrote", rej_csv, "n=%d" % len(reject_info))

    coords = export_results(kept, OUT_DIR, pixel_um=PIXEL_UM, dz_um=PSF_DZ_UM)
    mean_imgs = mean_images_from_memmaps(mems, move)

    plot_dir = OUT_DIR / TRACE_PLOT_DIR_NAME
    if plot_dir.is_dir():
        for old in plot_dir.glob("neuron_*_traces.png"):
            old.unlink()

    print("\n=== trace figures for KEPT (T stride=%d) ===" % TRACE_T_STRIDE)
    save_neuron_trace_plots(
        kept,
        OUT_DIR,
        mems=mems,
        move=move,
        mean_imgs=mean_imgs,
        t_stride=TRACE_T_STRIDE,
    )
    if coords.size:
        print(
            "kept depth um: mean=%.2f  std=%.2f  n=%d"
            % (float(coords[:, 2].mean()), float(coords[:, 2].std()), len(kept))
        )
    print("Done. kept=%d rejected=%d" % (len(kept), len(rejected)))


def plot_traces_from_saved_dictionary() -> None:
    """Export per-neuron traces from data.bin memmaps (full 6000 frames by default)."""
    dict_path = OUT_DIR / "neuron_dictionary.npy"
    if not dict_path.is_file():
        raise FileNotFoundError(dict_path)
    merged = list(np.load(str(dict_path), allow_pickle=True))
    print("loaded %d neurons from %s" % (len(merged), dict_path))

    coef_path = OUT_DIR / "psf_fit_coef.npz"
    data = np.load(str(coef_path))
    centers = np.asarray(data["centers_1based"], dtype=np.float64)
    move = compute_move(centers, factor=MOVE_DIVISOR)
    print("move=\n", move)

    print("opening data.bin memmaps (TRACE_T_STRIDE=%d) ..." % TRACE_T_STRIDE)
    mems, ly, lx, nf = open_view_memmaps(DATA_ROOT)
    print("FOV %dx%d, nframes=%d" % (ly, lx, nf))
    mean_imgs = mean_images_from_memmaps(mems, move)
    save_neuron_trace_plots(
        merged,
        OUT_DIR,
        mems=mems,
        move=move,
        mean_imgs=mean_imgs,
        t_stride=TRACE_T_STRIDE,
    )



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("DATA_ROOT =", DATA_ROOT)
    print("OUT_DIR   =", OUT_DIR)
    print(
        "FACTOR=%.3f  MOVE_DIVISOR=%.3f  PSF_DZ_UM=%.2f  T_STRIDE=%d  DIST_THR=%.1f"
        % (FACTOR, MOVE_DIVISOR, PSF_DZ_UM, T_STRIDE, DIST_THR)
    )

    print("\n=== 1/4 PSF calibration ===")
    coef_path = OUT_DIR / "psf_fit_coef.npz"
    if coef_path.is_file() and not FORCE_RECALIB:
        print("reusing existing", coef_path)
        data = np.load(str(coef_path))
        calib = {
            "coef_xz": np.asarray(data["coef_psf_xz"], dtype=np.float64),
            "coef_yz": np.asarray(data["coef_psf_yz"], dtype=np.float64),
            "x0": float(data["x0"]),
            "y0": float(data["y0"]),
            "z_um": np.asarray(data["z_um"], dtype=np.float64),
            "centers_1based": np.asarray(data["centers_1based"], dtype=np.float64),
        }
    else:
        if FORCE_RECALIB:
            print("FORCE_RECALIB: ignoring existing", coef_path)
        calib = calibrate_psf()
    coef_xz = calib["coef_xz"]
    coef_yz = calib["coef_yz"]
    x0, y0 = float(calib["x0"]), float(calib["y0"])
    centers = calib["centers_1based"]
    z_um = calib["z_um"]
    half_span = float(max(abs(z_um[0]), abs(z_um[-1])))
    n_steps = int(np.ceil(half_span / DEPTH_STEP_UM))
    depth_min = -n_steps * DEPTH_STEP_UM
    depth_max = n_steps * DEPTH_STEP_UM
    print(
        "depth search: [%d, %d] um step=%d (%d samples)"
        % (
            depth_min,
            depth_max,
            DEPTH_STEP_UM,
            len(range(depth_min, depth_max + 1, DEPTH_STEP_UM)),
        )
    )

    move = compute_move(centers, factor=MOVE_DIVISOR)
    # Prefer integer pixel shifts; if non-zero, round already done in compute_move
    print("move=\n", move)
    print("x0=%.2f y0=%.2f" % (x0, y0))
    arch = OUT_DIR / "archive_autoCenter" / "psf_fit_coef.npz"
    if arch.is_file():
        old = np.load(str(arch))
        print("archive auto-center xz slopes:", np.asarray(old["coef_psf_xz"])[0])
        print("archive auto-center yz slopes:", np.asarray(old["coef_psf_yz"])[0])
        print("archive centers_1based:\n", np.asarray(old["centers_1based"]))

    print("\n=== 2/4 load suite2p ===")
    s2p = []
    for v in range(1, 5):
        plane = view_plane_dir(v, DATA_ROOT)
        sp = load_suite2p_plane(plane)
        print(
            "view%d: ROIs=%d iscell=%d"
            % (v, len(sp["stat"]), sp["Order_neuron"].size)
        )
        s2p.append(sp)

    print("\n=== 3/4 load movies (block-average bin=%d) ===" % T_STRIDE)
    views = load_views_from_bin(DATA_ROOT, t_stride=T_STRIDE)
    print("views shape (Y,X,T,V)=", views.shape)
    print("approx RAM GB=%.2f" % (views.nbytes / 1e9))
    move_for_plots = np.asarray(move, dtype=np.float64).copy()
    move = apply_integer_move_inplace(views, move, s2p)
    print("applied integer move; estimate_depth move now zero")

    print("\n=== 4/4 localize + merge ===")
    dicts, _peak_corrs = localize_all_views(
        views,
        s2p,
        coef_xz,
        coef_yz,
        move,
        factor=FACTOR,
        x0=x0,
        y0=y0,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    n_t = views.shape[2]
    repair_traces_from_F(dicts, s2p, t_stride=T_STRIDE, n_t=n_t)

    merged = merge_dictionary1(
        dicts[0], dicts[1], dicts[2], dicts[3],
        dist_thr=DIST_THR,
        corr_thr=CORR_THR,
    )
    print(
        "merged neurons=%d (per-view: %s)"
        % (len(merged), [len(d) for d in dicts])
    )
    np.save(str(OUT_DIR / "neuron_dictionary_all.npy"), np.array(merged, dtype=object))

    del views
    print("\n=== quality filter (full traces from data.bin) ===")
    mems, ly, lx, nf = open_view_memmaps(DATA_ROOT)
    print("FOV %dx%d, nframes=%d" % (ly, lx, nf))
    # Use localization stride for fast filter pass, then full T for kept plots
    traces_for_filter = extract_all_traces_memmap(
        merged, mems, move_for_plots, t_stride=max(T_STRIDE, 1)
    )
    kept, rejected, reject_info = filter_merged_neurons(merged, traces_for_filter)

    # write reject log
    rej_csv = OUT_DIR / "localization_rejected.csv"
    with open(rej_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "orig_idx",
            "peak_corr",
            "center_spread_px",
            "mean_trace_corr",
            "min_trace_corr",
            "best_center_views",
            "best_corr_views",
            "reasons",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(reject_info)
    print("wrote", rej_csv, "n=%d" % len(reject_info))

    coords = export_results(kept, OUT_DIR, pixel_um=PIXEL_UM, dz_um=PSF_DZ_UM)
    mean_imgs = mean_images_from_memmaps(mems, move_for_plots)

    # clear old trace figures so rejected don't linger
    plot_dir = OUT_DIR / TRACE_PLOT_DIR_NAME
    if plot_dir.is_dir():
        for old in plot_dir.glob("neuron_*_traces.png"):
            old.unlink()

    print("\n=== 4-view trace figures for KEPT neurons (T stride=%d) ===" % TRACE_T_STRIDE)
    save_neuron_trace_plots(
        kept,
        OUT_DIR,
        mems=mems,
        move=move_for_plots,
        mean_imgs=mean_imgs,
        t_stride=TRACE_T_STRIDE,
    )

    if coords.size:
        print(
            "\nkept depth um: mean=%.2f  std=%.2f  min=%.2f  max=%.2f"
            % (
                float(coords[:, 2].mean()),
                float(coords[:, 2].std()),
                float(coords[:, 2].min()),
                float(coords[:, 2].max()),
            )
        )
    print("\nDone. kept=%d rejected=%d  Outputs in %s" % (len(kept), len(rejected), OUT_DIR))


if __name__ == "__main__":
    if REFILTER_ONLY:
        refilter_from_saved_all()
    elif PLOT_TRACES_ONLY:
        plot_traces_from_saved_dictionary()
    else:
        main()
