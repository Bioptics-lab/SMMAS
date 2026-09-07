# -*- coding: utf-8 -*-
"""Recover 3D positions for method2 ROIs via 4-view data.bin parallax search.

Pipeline
--------
1. Read matches from CA1R1f5Output/match/loose/ (fallback: baseline/).
2. Trusted matches (strict gates) keep method1 (col,row,z). This includes
   pairs added by promote_unmatched_via_view_suite2p.py after the first match.
3. Weak + unmatched m2: seed by inverse-XY warp, pick best primary at z2,
   sweep z (and XY refine) correlating disk/warped-footprint traces
   against m2 F (time-strided for speed).
4. Merge into CA1R1f5Output/match/m2_3d_coverage/.

Originals (suite2p, localization, dictionary, psf) are read-only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from dataset_config import move_divisor_from_coef, view_plane_dir
from tplfm_utils import compute_move as _compute_move, corrcoef_scalar, load_npy, parallax_dx_dy

from match_linear_runner_to_recon_planes import (
    M1_DATA,
    M1_OUT,
    MATCH_ROOT,
    load_matches_csv,
    load_method1,
    load_method2,
)

# Prefer loose rematch; fall back to baseline
MATCH_LOOSE = MATCH_ROOT / "loose"
MATCH_BASE = MATCH_ROOT / "baseline"
MATCH_DIR = MATCH_LOOSE if (MATCH_LOOSE / "matches.csv").is_file() else MATCH_BASE

COV_DIR = MATCH_ROOT / "m2_3d_coverage"
REC_DIR = COV_DIR / "recovered_from_bin"
PLOT_DIR = REC_DIR / "pair_plots"

# Trust gates (keep method1 3D; do not bin-recover these m2)
TRUST_R_MIN = 0.40
TRUST_D_XY_PX = 20.0
TRUST_D_Z_UM = 12.0

# Bin recovery search
DISK_RADIUS = 5
Z_SEARCH_HALF = 20.0
Z_STEP = 2.0
XY_COARSE_HALF = 9   # px, step 3 — absorbs registration/frame-frame offsets
XY_FINE_HALF = 2     # px, step 1 around coarse best
T_STRIDE = 15
R_BEST = 0.22
R_PASS = 0.15
N_PASS = 2


def resolve_match_dir() -> Path:
    if (MATCH_LOOSE / "matches.csv").is_file():
        return MATCH_LOOSE
    if (MATCH_BASE / "matches.csv").is_file():
        return MATCH_BASE
    raise FileNotFoundError(
        "no matches.csv in match/loose/ or match/baseline/"
    )


def load_xy_tfm(match_dir: Path) -> dict[str, float]:
    p = match_dir / "xy_transform.npz"
    if not p.is_file():
        p = MATCH_BASE / "xy_transform.npz"
    z = np.load(str(p))
    return {
        "s": float(z["s"]),
        "tx": float(z["tx"]),
        "ty": float(z["ty"]),
        "ncc": float(z["ncc"]) if "ncc" in z.files else float("nan"),
    }


def inverse_xy(
    x2: float | np.ndarray, y2: float | np.ndarray, tfm: dict[str, float]
) -> tuple[Any, Any]:
    s, tx, ty = float(tfm["s"]), float(tfm["tx"]), float(tfm["ty"])
    return (x2 - tx) / s, (y2 - ty) / s


def open_memmaps() -> tuple[list, int, int, int]:
    mems = []
    ly = lx = nf = None
    for v in range(1, 5):
        plane = view_plane_dir(v, M1_DATA)
        ops = load_npy(plane / "ops.npy", allow_pickle=True).item()
        ly_v, lx_v, nf_v = int(ops["Ly"]), int(ops["Lx"]), int(ops["nframes"])
        if ly is None:
            ly, lx, nf = ly_v, lx_v, nf_v
        bin_path = plane / "data.bin"
        dtype = np.int16 if bin_path.stat().st_size == nf_v * ly_v * lx_v * 2 else np.float32
        mems.append(np.memmap(str(bin_path), dtype=dtype, mode="r", shape=(nf_v, ly_v, lx_v)))
    assert ly is not None and lx is not None and nf is not None
    return mems, ly, lx, nf


def disk_pixels(cx: float, cy: float, radius: int, h: int, w: int):
    r = int(max(1, radius))
    xs, ys = [], []
    rr2 = float(r * r)
    for yi in range(int(np.floor(cy - r)), int(np.ceil(cy + r)) + 1):
        for xi in range(int(np.floor(cx - r)), int(np.ceil(cx + r)) + 1):
            if 0 <= xi < w and 0 <= yi < h and (xi - cx) ** 2 + (yi - cy) ** 2 <= rr2:
                xs.append(xi)
                ys.append(yi)
    if not xs:
        return (
            np.array([int(np.clip(round(cx), 0, w - 1))], dtype=np.int32),
            np.array([int(np.clip(round(cy), 0, h - 1))], dtype=np.int32),
        )
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)


def warped_m2_footprint(
    m2: dict[str, Any], tfm: dict[str, float], h: int, w: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Inverse-warp m2 xpix/ypix into method1 FOV; None if empty/invalid."""
    xpix = np.asarray(m2.get("xpix", []), dtype=np.float64).ravel()
    ypix = np.asarray(m2.get("ypix", []), dtype=np.float64).ravel()
    if xpix.size < 3:
        return None
    col, row = inverse_xy(xpix, ypix, tfm)
    xs = np.rint(col).astype(int)
    ys = np.rint(row).astype(int)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[ok], ys[ok]
    if xs.size < 3:
        return None
    stacked = np.unique(np.column_stack([xs, ys]), axis=0)
    return stacked[:, 0].astype(np.int32), stacked[:, 1].astype(np.int32)


def footprint_at_center(
    seed_xs: np.ndarray | None,
    seed_ys: np.ndarray | None,
    cx: float,
    cy: float,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place seed footprint deltas at (cx,cy), or disk fallback."""
    if seed_xs is not None and seed_ys is not None and seed_xs.size:
        scx, scy = float(seed_xs.mean()), float(seed_ys.mean())
        xs = np.rint(seed_xs - scx + cx).astype(int)
        ys = np.rint(seed_ys - scy + cy).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[ok], ys[ok]
        if xs.size >= 3:
            return xs.astype(np.int32), ys.astype(np.int32)
    return disk_pixels(cx, cy, DISK_RADIUS, h, w)


# ---------------------------------------------------------------------------
# Per-candidate view crops: load each view's search box ONCE from the memmap
# ---------------------------------------------------------------------------
class CropViews:
    """(n_t, bh, bw) float32 crops per view + origin; answers trace queries.

    Only the time indices in ``frame_idx`` are copied from the memmap (not full T).
    """

    def __init__(
        self,
        mems,
        boxes: list[tuple[int, int, int, int]],
        frame_idx: np.ndarray,
    ) -> None:
        self.crops: list[np.ndarray] = []
        self.origins: list[tuple[int, int]] = []  # (y0, x0)
        idx = np.asarray(frame_idx, dtype=np.int64)
        for v, mem in enumerate(mems):
            y0, y1, x0, x1 = boxes[v]
            crop = np.asarray(mem[idx, y0:y1, x0:x1], dtype=np.float32)
            self.crops.append(crop)
            self.origins.append((y0, x0))

    def trace(self, v: int, xs_abs: np.ndarray, ys_abs: np.ndarray) -> np.ndarray:
        n = int(self.crops[v].shape[0])
        y0, x0 = self.origins[v]
        bh, bw = self.crops[v].shape[1], self.crops[v].shape[2]
        xs = np.asarray(xs_abs, dtype=np.int64) - x0
        ys = np.asarray(ys_abs, dtype=np.int64) - y0
        ok = (xs >= 0) & (xs < bw) & (ys >= 0) & (ys < bh)
        if int(ok.sum()) < 3:
            return np.zeros(n, dtype=np.float64)
        xs, ys = xs[ok], ys[ok]
        if xs.size > 80:  # cap footprint for speed
            sel = np.linspace(0, xs.size - 1, 80).astype(int)
            xs, ys = xs[sel], ys[sel]
        block = self.crops[v][:, ys, xs]  # (n, npix)
        return block.mean(axis=1, dtype=np.float64)


def view_center_in_bin(
    col, row, z, primary_v0, view_v0, move_i, coef_xz, coef_yz, factor, x0, y0
):
    dx_p, dy_p = parallax_dx_dy(z, primary_v0, coef_xz, coef_yz, factor, x0, y0)
    dx_v, dy_v = parallax_dx_dy(z, view_v0, coef_xz, coef_yz, factor, x0, y0)
    cx = col + (dx_v - dx_p) + float(move_i[view_v0, 1])
    cy = row + (dy_v - dy_p) + float(move_i[view_v0, 0])
    return cx, cy


def search_boxes(
    col0, row0, z2, move_i, coef_xz, coef_yz, factor, x0, y0, ly, lx,
    seed_fp,
) -> list[tuple[int, int, int, int]]:
    """Per-view (y0,y1,x0,x1) box covering the full z / xy / primary search."""
    if seed_fp is not None:
        fp_hx = float(np.max(np.abs(seed_fp[0] - seed_fp[0].mean())))
        fp_hy = float(np.max(np.abs(seed_fp[1] - seed_fp[1].mean())))
    else:
        fp_hx = fp_hy = float(DISK_RADIUS)
    pad = max(fp_hx, fp_hy) + XY_COARSE_HALF + 3.0
    boxes = []
    for v0 in range(4):
        cx_min, cx_max = np.inf, -np.inf
        cy_min, cy_max = np.inf, -np.inf
        for z in (z2 - Z_SEARCH_HALF, z2 + Z_SEARCH_HALF):
            for p0 in range(4):
                cx, cy = view_center_in_bin(
                    col0, row0, z, p0, v0, move_i, coef_xz, coef_yz, factor, x0, y0
                )
                cx_min, cx_max = min(cx_min, cx), max(cx_max, cx)
                cy_min, cy_max = min(cy_min, cy), max(cy_max, cy)
        x0b = int(max(0, np.floor(cx_min - pad)))
        x1b = int(min(lx, np.ceil(cx_max + pad + 1)))
        y0b = int(max(0, np.floor(cy_min - pad)))
        y1b = int(min(ly, np.ceil(cy_max + pad + 1)))
        boxes.append((y0b, y1b, x0b, x1b))
    return boxes


def eval_views_at(
    col: float,
    row: float,
    z: float,
    p0: int,
    crops: CropViews,
    ly: int,
    lx: int,
    frame_idx: np.ndarray,
    tr_m2: np.ndarray,
    move_i,
    coef_xz,
    coef_yz,
    factor,
    x0,
    y0,
    seed_fp: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (rs[4], traces[4,n], rmax, score=sum of positive r)."""
    n = frame_idx.size
    rs = np.full(4, np.nan)
    traces = np.zeros((4, n), dtype=np.float64)
    seed_xs = seed_fp[0] if seed_fp else None
    seed_ys = seed_fp[1] if seed_fp else None
    for v0 in range(4):
        cx, cy = view_center_in_bin(
            col, row, z, p0, v0, move_i, coef_xz, coef_yz, factor, x0, y0
        )
        xs, ys = footprint_at_center(seed_xs, seed_ys, cx, cy, ly, lx)
        traces[v0] = crops.trace(v0, xs, ys)
        r = corrcoef_scalar(traces[v0], tr_m2)
        if np.isfinite(r):
            rs[v0] = r
    rmax = float(np.nanmax(rs)) if np.any(np.isfinite(rs)) else -np.inf
    score = float(np.sum(np.clip(np.nan_to_num(rs, nan=0.0), 0.0, None)))
    return rs, traces, rmax, score


def recover_one(
    m2: dict[str, Any],
    tfm: dict[str, float],
    mems,
    ly: int,
    lx: int,
    nf: int,
    move_i,
    coef_xz,
    coef_yz,
    factor: float,
    x0: float,
    y0: float,
) -> dict[str, Any]:
    x2, y2, z2 = float(m2["x"]), float(m2["y"]), float(m2["z_um"])
    col0, row0 = inverse_xy(x2, y2, tfm)
    tr_full = np.asarray(m2["trace"], dtype=np.float64)
    n_t = min(nf, tr_full.shape[0])
    frame_idx = np.arange(0, n_t, T_STRIDE, dtype=int)
    tr_m2 = tr_full[frame_idx]

    seed_fp = warped_m2_footprint(m2, tfm, ly, lx)

    # Load one crop per view covering the entire search volume (the speed fix)
    boxes = search_boxes(
        col0, row0, z2, move_i, coef_xz, coef_yz, factor, x0, y0, ly, lx, seed_fp
    )
    crops = CropViews(mems, boxes, frame_idx)

    args = (crops, ly, lx, frame_idx, tr_m2, move_i, coef_xz, coef_yz, factor, x0, y0, seed_fp)

    # 1) pick best primary at plane z2 by 4-view positive-sum score
    best_p0, best_score = 0, -np.inf
    for p0 in range(4):
        _rs, _tr, _rmax, score = eval_views_at(col0, row0, z2, p0, *args)
        if score > best_score:
            best_p0, best_score = p0, score
    p0 = int(best_p0)

    # 2) sweep z (all 4 views, positive-sum score)
    z_grid = np.arange(z2 - Z_SEARCH_HALF, z2 + Z_SEARCH_HALF + Z_STEP / 2, Z_STEP)
    depth_curve: list[tuple[float, float]] = []
    best_z, best_zs = float(z2), -np.inf
    for z in z_grid:
        _rs, _tr, _rmax, score = eval_views_at(col0, row0, float(z), p0, *args)
        depth_curve.append((float(z), float(score)))
        if score > best_zs:
            best_zs, best_z = score, float(z)

    # 3) XY refine at best_z: coarse +/-9 step 3, then fine +/-2 step 1
    best_col, best_row = float(col0), float(row0)
    best_xy_score = best_zs
    coarse = range(-XY_COARSE_HALF, XY_COARSE_HALF + 1, 3)
    for dcol in coarse:
        for drow in coarse:
            if dcol == 0 and drow == 0:
                continue
            _rs, _tr, _rmax, score = eval_views_at(
                col0 + dcol, row0 + drow, best_z, p0, *args
            )
            if score > best_xy_score:
                best_xy_score = score
                best_col, best_row = float(col0 + dcol), float(row0 + drow)
    bc0, br0 = best_col, best_row
    for dcol in range(-XY_FINE_HALF, XY_FINE_HALF + 1):
        for drow in range(-XY_FINE_HALF, XY_FINE_HALF + 1):
            if dcol == 0 and drow == 0:
                continue
            _rs, _tr, _rmax, score = eval_views_at(
                bc0 + dcol, br0 + drow, best_z, p0, *args
            )
            if score > best_xy_score:
                best_xy_score = score
                best_col, best_row = float(bc0 + dcol), float(br0 + drow)

    # 4) final 4-view eval at refined pose
    rs, traces, rmax, score = eval_views_at(best_col, best_row, best_z, p0, *args)
    n_pass = int(np.sum(np.isfinite(rs) & (rs >= R_PASS)))
    accepted = bool(np.isfinite(rmax) and rmax >= R_BEST and n_pass >= N_PASS)
    del crops

    return {
        "m2": m2,
        "col": best_col,
        "row": best_row,
        "z2": z2,
        "best_r": float(rmax),
        "best_z": best_z,
        "primary_v0": p0,
        "rs": rs,
        "n_pass": n_pass,
        "accepted": accepted,
        "traces": traces,
        "frame_idx": frame_idx,
        "tr_m2_stride": tr_m2,
        "depth_curve": depth_curve,
        "used_warped_fp": seed_fp is not None,
    }


def build_synthetic_entry(res, coef_xz, coef_yz, factor, x0, y0) -> dict[str, Any]:
    p0 = int(res["primary_v0"])
    z = float(res["best_z"])
    col, row = float(res["col"]), float(res["row"])
    dx_p, dy_p = parallax_dx_dy(z, p0, coef_xz, coef_yz, factor, x0, y0)

    center_allview = np.zeros((2, 4), dtype=np.float64)
    center_allview[0, p0] = col
    center_allview[1, p0] = row
    for v0 in range(4):
        if v0 == p0:
            continue
        dx_v, dy_v = parallax_dx_dy(z, v0, coef_xz, coef_yz, factor, x0, y0)
        center_allview[0, v0] = col + (dx_v - dx_p)
        center_allview[1, v0] = row + (dy_v - dy_p)

    xs, ys = disk_pixels(col, row, DISK_RADIUS, 10_000, 10_000)
    return {
        "center": np.array([col, row, z], dtype=np.float64),
        "center_allview": center_allview,
        "pixels1": (xs - col).astype(np.float64),
        "pixels2": (ys - row).astype(np.float64),
        "primary_view": int(p0 + 1),
        "peak_corr": float(res["best_r"]),
        "source": "recovered_from_bin",
        "m2_plane": int(res["m2"]["plane_id"]),
        "m2_roi": int(res["m2"]["roi_idx"]),
        "recovery_rs": np.asarray(res["rs"], dtype=np.float64).copy(),
        "trace_primary": np.asarray(res["traces"][p0], dtype=np.float64).copy(),
    }


def plot_one_recovery(res: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={"height_ratios": [1.0, 1.2]})
    ax0, ax1 = axes
    zz = [p[0] for p in res["depth_curve"]]
    rv = [p[1] for p in res["depth_curve"]]
    ax0.plot(zz, rv, "o-", ms=3, color="C0")
    ax0.axvline(res["z2"], color="C1", ls="--", lw=1, label="m2 z")
    ax0.axvline(res["best_z"], color="C2", ls="--", lw=1, label="best z")
    ax0.set_xlabel("z (um)")
    ax0.set_ylabel("sum of positive view r")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=7, loc="best")
    ax0.set_title(
        "p%d/roi%d  best_r=%.3f  z=%.1f  primary=v%d  %s"
        % (
            res["m2"]["plane_id"],
            res["m2"]["roi_idx"],
            res["best_r"],
            res["best_z"],
            res["primary_v0"] + 1,
            "ACCEPT" if res["accepted"] else "reject",
        ),
        fontsize=10,
    )

    t = res["frame_idx"]
    tr_m2 = res["tr_m2_stride"]
    p0 = res["primary_v0"]
    tr_p = res["traces"][p0]

    def _z(a):
        a = np.asarray(a, dtype=np.float64)
        s = float(a.std())
        return (a - a.mean()) / s if s > 1e-12 else np.zeros_like(a)

    ax1.plot(t, _z(tr_m2), color="C0", lw=0.55, label="m2 F", alpha=0.9)
    ax1.plot(t, _z(tr_p), color="C3", lw=0.55, label="bin primary", alpha=0.85)
    ax1.set_xlabel("frame (stride=%d)" % T_STRIDE)
    ax1.set_ylabel("z-score")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=95)
    except OSError as exc:  # plotting must never kill the pipeline
        print(f"      [warn] plot save failed: {exc}")
    plt.close(fig)


def classify_matches(
    matches: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trusted, weak = [], []
    for m in matches:
        ok = (
            float(m["corr"]) >= TRUST_R_MIN
            and float(m["d_xy_px"]) <= TRUST_D_XY_PX
            and float(m["d_z_um"]) <= TRUST_D_Z_UM
        )
        (trusted if ok else weak).append(m)
    return trusted, weak


def merge_catalog(
    m2_pool: list[dict[str, Any]],
    trusted: list[dict[str, Any]],
    recover_results: list[dict[str, Any]],
    entries_m1: list[dict[str, Any]],
    c1_orig: np.ndarray,
    primaries: np.ndarray,
    recovered_entries: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trust_map = {(int(m["m2_plane"]), int(m["m2_roi"])): m for m in trusted}
    rec_map = {
        (int(r["m2"]["plane_id"]), int(r["m2"]["roi_idx"])): r for r in recover_results
    }
    rec_entry_map = {
        (int(e["m2_plane"], ), int(e["m2_roi"])): e for e in recovered_entries
    }

    rows = []
    dict_out: list[dict[str, Any]] = []
    failed = []
    by_src: dict[str, int] = {}
    by_plane_ok: dict[int, int] = {}
    by_plane_tot: dict[int, int] = {}

    for m2 in m2_pool:
        key = (int(m2["plane_id"]), int(m2["roi_idx"]))
        by_plane_tot[key[0]] = by_plane_tot.get(key[0], 0) + 1
        if key in trust_map:
            m = trust_map[key]
            i1 = int(m["m1_idx"])
            src = "matched_trusted"
            col, row, z = float(c1_orig[i1, 0]), float(c1_orig[i1, 1]), float(c1_orig[i1, 2])
            corr = float(m["corr"])
            pv = int(primaries[i1])
            entry = dict(entries_m1[i1])
            entry = {
                "center": np.array([col, row, z], dtype=np.float64),
                "center_allview": np.asarray(
                    entry.get("center_allview", np.zeros((2, 4))), dtype=np.float64
                ).copy(),
                "pixels1": np.asarray(entry.get("pixels1", []), dtype=np.float64).copy(),
                "pixels2": np.asarray(entry.get("pixels2", []), dtype=np.float64).copy(),
                "primary_view": pv,
                "peak_corr": corr,
                "source": src,
                "m2_plane": key[0],
                "m2_roi": key[1],
                "m1_idx": i1,
            }
            rows.append(
                {
                    "m2_plane": key[0],
                    "m2_roi": key[1],
                    "source": src,
                    "col": col,
                    "row": row,
                    "z_um": z,
                    "corr_or_best_r": corr,
                    "primary_view": pv,
                    "m1_idx": i1,
                    "d_xy_px": float(m["d_xy_px"]),
                    "d_z_um": float(m["d_z_um"]),
                    "n_pass": "",
                    "accepted": 1,
                }
            )
            dict_out.append(entry)
            by_plane_ok[key[0]] = by_plane_ok.get(key[0], 0) + 1
        elif key in rec_map and rec_map[key]["accepted"] and key in rec_entry_map:
            r = rec_map[key]
            e = rec_entry_map[key]
            src = "recovered_bin"
            rows.append(
                {
                    "m2_plane": key[0],
                    "m2_roi": key[1],
                    "source": src,
                    "col": float(r["col"]),
                    "row": float(r["row"]),
                    "z_um": float(r["best_z"]),
                    "corr_or_best_r": float(r["best_r"]),
                    "primary_view": int(r["primary_v0"] + 1),
                    "m1_idx": -1,
                    "d_xy_px": "",
                    "d_z_um": abs(float(r["best_z"]) - float(r["z2"])),
                    "n_pass": int(r["n_pass"]),
                    "accepted": 1,
                }
            )
            dict_out.append(e)
            by_plane_ok[key[0]] = by_plane_ok.get(key[0], 0) + 1
        else:
            src = "failed"
            r = rec_map.get(key)
            rows.append(
                {
                    "m2_plane": key[0],
                    "m2_roi": key[1],
                    "source": src,
                    "col": "",
                    "row": "",
                    "z_um": "",
                    "corr_or_best_r": float(r["best_r"]) if r else "",
                    "primary_view": "",
                    "m1_idx": -1,
                    "d_xy_px": "",
                    "d_z_um": "",
                    "n_pass": int(r["n_pass"]) if r else "",
                    "accepted": 0,
                }
            )
            failed.append({"m2_plane": key[0], "m2_roi": key[1],
                           "best_r": float(r["best_r"]) if r else None})
        by_src[src] = by_src.get(src, 0) + 1

    fields = list(rows[0].keys()) if rows else []
    with open(out_dir / "m2_3d_catalog.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    np.save(str(out_dir / "m2_3d_dictionary.npy"), np.array(dict_out, dtype=object))

    n_ok = sum(1 for r in rows if r["source"] != "failed")
    summary = {
        "n_m2": len(m2_pool),
        "n_with_3d": n_ok,
        "coverage": float(n_ok) / max(1, len(m2_pool)),
        "by_source": by_src,
        "by_plane_ok": by_plane_ok,
        "by_plane_tot": by_plane_tot,
        "n_failed": len(failed),
        "failed": failed,
        "match_dir": str(MATCH_DIR),
        "trust_gates": {
            "TRUST_R_MIN": TRUST_R_MIN,
            "TRUST_D_XY_PX": TRUST_D_XY_PX,
            "TRUST_D_Z_UM": TRUST_D_Z_UM,
        },
        "recover_gates": {
            "R_BEST": R_BEST,
            "R_PASS": R_PASS,
            "N_PASS": N_PASS,
            "T_STRIDE": T_STRIDE,
            "Z_SEARCH_HALF": Z_SEARCH_HALF,
            "XY_COARSE_HALF": XY_COARSE_HALF,
            "XY_FINE_HALF": XY_FINE_HALF,
        },
        "originals_untouched": True,
        "output_dir": str(out_dir),
    }
    with open(out_dir / "coverage_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        "coverage: %d/%d (%.1f%%)  by_source=%s"
        % (n_ok, len(m2_pool), 100.0 * n_ok / max(1, len(m2_pool)), by_src)
    )
    print("wrote", out_dir / "m2_3d_catalog.csv")
    print("wrote", out_dir / "m2_3d_dictionary.npy", "n=%d" % len(dict_out))
    print("wrote", out_dir / "coverage_summary.json")


_REPORT_FIELDS = [
    "m2_plane", "m2_roi", "col", "row", "z2", "best_z", "best_r",
    "r_v1", "r_v2", "r_v3", "r_v4", "n_pass", "accepted", "primary_view",
    "used_warped_fp",
]


def _write_report(results: list[dict[str, Any]]) -> None:
    """Checkpoint the recovery report so a mid-run abort loses nothing."""
    REC_DIR.mkdir(parents=True, exist_ok=True)
    with open(REC_DIR / "recovery_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_REPORT_FIELDS)
        w.writeheader()
        for r in results:
            rs = r["rs"]
            w.writerow(
                {
                    "m2_plane": int(r["m2"]["plane_id"]),
                    "m2_roi": int(r["m2"]["roi_idx"]),
                    "col": r["col"],
                    "row": r["row"],
                    "z2": r["z2"],
                    "best_z": r["best_z"],
                    "best_r": r["best_r"],
                    "r_v1": rs[0],
                    "r_v2": rs[1],
                    "r_v3": rs[2],
                    "r_v4": rs[3],
                    "n_pass": r["n_pass"],
                    "accepted": int(r["accepted"]),
                    "primary_view": int(r["primary_v0"] + 1),
                    "used_warped_fp": int(r["used_warped_fp"]),
                }
            )


def main() -> None:
    global MATCH_DIR
    MATCH_DIR = resolve_match_dir()
    COV_DIR.mkdir(parents=True, exist_ok=True)
    REC_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("MATCH_DIR =", MATCH_DIR)
    print("COV_DIR   =", COV_DIR)
    print(
        "trust: R>=%.2f dxy<=%.1f dz<=%.1f | recover: R_BEST=%.2f R_PASS=%.2f N_PASS=%d stride=%d"
        % (TRUST_R_MIN, TRUST_D_XY_PX, TRUST_D_Z_UM, R_BEST, R_PASS, N_PASS, T_STRIDE)
    )

    matches = load_matches_csv(MATCH_DIR / "matches.csv")
    trusted, weak = classify_matches(matches)
    um2_path = MATCH_DIR / "unmatched_m2.csv"
    um2_rows = list(csv.DictReader(open(um2_path, encoding="utf-8"))) if um2_path.is_file() else []
    print(
        "matches=%d trusted=%d weak=%d unmatched_m2=%d"
        % (len(matches), len(trusted), len(weak), len(um2_rows))
    )

    print("\n=== load method1 / method2 ===")
    entries_m1, c1_orig, primaries = load_method1()
    m2_pool = load_method2()
    m2_map = {(m["plane_id"], m["roi_idx"]): i for i, m in enumerate(m2_pool)}

    # candidates: weak match m2 + unmatched m2 + any m2 not trusted
    cand_keys: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for m in weak:
        key = (int(m["m2_plane"]), int(m["m2_roi"]))
        if key not in seen:
            cand_keys.append(key)
            seen.add(key)
    for u in um2_rows:
        key = (int(u["m2_plane"]), int(u["m2_roi"]))
        if key not in seen:
            cand_keys.append(key)
            seen.add(key)
    trust_keys = {(int(m["m2_plane"]), int(m["m2_roi"])) for m in trusted}
    for m2 in m2_pool:
        key = (int(m2["plane_id"]), int(m2["roi_idx"]))
        if key not in trust_keys and key not in seen:
            cand_keys.append(key)
            seen.add(key)
    print("bin-recover candidates: %d" % len(cand_keys))

    tfm = load_xy_tfm(MATCH_DIR)
    print("XY tfm: s=%.4f tx=%.2f ty=%.2f" % (tfm["s"], tfm["tx"], tfm["ty"]))

    coef = np.load(str(M1_OUT / "psf_fit_coef.npz"))
    coef_xz = np.asarray(coef["coef_psf_xz"], dtype=np.float64)
    coef_yz = np.asarray(coef["coef_psf_yz"], dtype=np.float64)
    factor = float(coef["factor"]) if "factor" in coef.files else 2.5
    x0 = float(coef["x0"]) if "x0" in coef.files else 60.0
    y0 = float(coef["y0"]) if "y0" in coef.files else 60.0
    move_div = move_divisor_from_coef(coef)
    move_i = np.round(
        _compute_move(
            np.asarray(coef["centers_1based"], dtype=np.float64), factor=move_div
        )
    ).astype(int)
    print("move_i=\n", move_i)

    print("\n=== open data.bin memmaps ===")
    mems, ly, lx, nf = open_memmaps()
    print("FOV %dx%d T=%d stride=%d" % (ly, lx, nf, T_STRIDE))

    results: list[dict[str, Any]] = []
    for k, key in enumerate(cand_keys):
        if key not in m2_map:
            print("  skip p%d/roi%d (not in pool)" % key)
            continue
        m2 = m2_pool[m2_map[key]]
        print(
            "  [%d/%d] p%d/roi%d ..."
            % (k + 1, len(cand_keys), key[0], key[1]),
            flush=True,
        )
        res = recover_one(
            m2, tfm, mems, ly, lx, nf, move_i, coef_xz, coef_yz, factor, x0, y0
        )
        print(
            "      best_r=%.3f z=%.1f col=%.1f row=%.1f primary=v%d n_pass=%d fp=%s -> %s"
            % (
                res["best_r"],
                res["best_z"],
                res["col"],
                res["row"],
                res["primary_v0"] + 1,
                res["n_pass"],
                "warp" if res["used_warped_fp"] else "disk",
                "ACCEPT" if res["accepted"] else "reject",
            ),
            flush=True,
        )
        results.append(res)
        _write_report(results)  # checkpoint after every candidate
        try:
            plot_one_recovery(
                res,
                PLOT_DIR
                / (
                    "rec_p%d_roi%03d_%s.png"
                    % (key[0], key[1], "ok" if res["accepted"] else "fail")
                ),
            )
        except Exception as exc:  # diagnostics only; never abort the run
            print("      [warn] plot failed: %s" % exc, flush=True)

    n_acc = sum(1 for r in results if r["accepted"])
    print("\naccepted recoveries %d / %d" % (n_acc, len(results)))
    _write_report(results)
    print("wrote", REC_DIR / "recovery_report.csv")

    recovered_entries = [
        build_synthetic_entry(r, coef_xz, coef_yz, factor, x0, y0)
        for r in results
        if r["accepted"]
    ]
    np.save(str(REC_DIR / "recovered_neurons.npy"), np.array(recovered_entries, dtype=object))
    print("wrote", REC_DIR / "recovered_neurons.npy", "n=%d" % len(recovered_entries))

    print("\n=== merge m2 3D catalog ===")
    merge_catalog(
        m2_pool,
        trusted,
        results,
        entries_m1,
        c1_orig,
        primaries,
        recovered_entries,
        COV_DIR,
    )
    print("\nDone. Originals untouched.")


if __name__ == "__main__":
    main()
