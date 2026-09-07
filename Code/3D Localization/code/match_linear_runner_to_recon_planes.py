# -*- coding: utf-8 -*-
"""Match method1 neurons to reconstruction-plane suite2p (method2).

Method1: 256 FOV multi-view localization (col, row, z_um) + primary-view traces
Method2: Planes/Plane{14,19,24,29} suite2p (255 FOV), z = -15/-5/+5/+15 um
         (2 um * (plane_id - 21.5); optional global sign flip via PLANE_Z_FLIPPED).

Cost = w_xy*dxy + w_z*|dz| + w_r*(1-r), gated by D_XY / D_Z / R_MIN.
Hungarian one-to-one assignment.

Outputs under CA1R1f5Output/match/{baseline|loose|zswap}/.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom as ndi_zoom
from scipy.optimize import linear_sum_assignment

from dataset_config import (
    DATA_ROOT,
    MATCH_MID_PLANE,
    MOVE_DIVISOR,
    OUT_DIR as M1_OUT,
    PLANE_Z_BASE,
    TRACE_T_STRIDE,
    plane_suite2p_dir,
    view_plane_dir,
)
from tplfm_utils import corrcoef_scalar, load_npy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
M1_DATA = DATA_ROOT
MATCH_ROOT = M1_OUT / "match"

# MATCH_MODE: "baseline" | "loose" | "zswap"
MATCH_MODE = "loose"
# True: negate all plane z (A/B test; set by refine_z_from_footprints.py)
PLANE_Z_FLIPPED = False


def plane_z_map() -> dict[int, float]:
    if PLANE_Z_FLIPPED or MATCH_MODE == "zswap":
        return {k: -v for k, v in PLANE_Z_BASE.items()}
    return dict(PLANE_Z_BASE)


PLANE_SPECS = [
    (plane_suite2p_dir(pid), pid, z)
    for pid, z in sorted(plane_z_map().items())
]

_GATES = {
    "baseline": dict(D_XY_PX=40.0, D_Z_UM=20.0, R_MIN=0.12),
    "loose": dict(D_XY_PX=50.0, D_Z_UM=24.0, R_MIN=0.08),
    "zswap": dict(D_XY_PX=40.0, D_Z_UM=20.0, R_MIN=0.12),
}
if MATCH_MODE not in _GATES:
    raise ValueError("MATCH_MODE must be baseline|loose|zswap, got %r" % MATCH_MODE)
D_XY_PX = float(_GATES[MATCH_MODE]["D_XY_PX"])
D_Z_UM = float(_GATES[MATCH_MODE]["D_Z_UM"])
R_MIN = float(_GATES[MATCH_MODE]["R_MIN"])
OUT_DIR = MATCH_ROOT / MATCH_MODE
W_XY = 1.0
W_Z = 0.35
W_R = 20.0
BIG = 1e6

# If True: skip rematch; plot one figure per row in matches.csv
PLOT_MATCHES_ONLY = False
# Skip per-pair PNGs (faster for rematch A/B)
SKIP_PAIR_PLOTS = True
MATCH_PLOT_DIR = OUT_DIR / "match_pair_plots"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_method1() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Return (entries, centers Nx3 [col,row,z], primary_views N)."""
    csv_path = M1_OUT / "localization.csv"
    dict_path = M1_OUT / "neuron_dictionary.npy"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not dict_path.is_file():
        raise FileNotFoundError(dict_path)

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    dictionary = list(np.load(str(dict_path), allow_pickle=True))
    if len(rows) != len(dictionary):
        print(
            "warn: CSV n=%d != dictionary n=%d; aligning by CSV idx"
            % (len(rows), len(dictionary))
        )

    entries: list[dict[str, Any]] = []
    centers = np.zeros((len(rows), 3), dtype=np.float64)
    primaries = np.zeros(len(rows), dtype=int)
    for i, row in enumerate(rows):
        idx = int(row["idx"])
        d = dictionary[idx] if idx < len(dictionary) else dictionary[i]
        col = float(row["col"])
        row_y = float(row["row"])
        z = float(row["z_um"])
        pv = int(float(row.get("primary_view", d.get("primary_view", 1))))
        entries.append(d)
        centers[i] = (col, row_y, z)
        primaries[i] = pv
    print("method1: %d kept neurons" % len(entries))
    return entries, centers, primaries


def load_method2() -> list[dict[str, Any]]:
    """Pool iscell ROIs from three recon planes."""
    pool: list[dict[str, Any]] = []
    for plane_dir, plane_id, z_um in PLANE_SPECS:
        iscell = np.asarray(load_npy(plane_dir / "iscell.npy", allow_pickle=False))
        F = np.asarray(load_npy(plane_dir / "F.npy", allow_pickle=False), dtype=np.float64)
        stat = list(load_npy(plane_dir / "stat.npy", allow_pickle=True))
        ops = load_npy(plane_dir / "ops.npy", allow_pickle=True).item()
        if iscell.ndim == 2:
            cell_mask = iscell[:, 0] > 0.5
        else:
            cell_mask = iscell.astype(bool)
        n_cell = int(cell_mask.sum())
        print(
            "method2 plane%d z=%.1f: FOV %dx%d  iscell=%d/%d"
            % (plane_id, z_um, int(ops["Ly"]), int(ops["Lx"]), n_cell, len(stat))
        )
        for roi_i in np.flatnonzero(cell_mask):
            st = stat[int(roi_i)]
            xpix = np.asarray(st["xpix"], dtype=np.float64).ravel()
            ypix = np.asarray(st["ypix"], dtype=np.float64).ravel()
            pool.append(
                {
                    "plane_id": int(plane_id),
                    "roi_idx": int(roi_i),
                    "z_um": float(z_um),
                    "x": float(xpix.mean()),
                    "y": float(ypix.mean()),
                    "xpix": xpix.astype(np.int32, copy=False),
                    "ypix": ypix.astype(np.int32, copy=False),
                    "trace": F[int(roi_i)].astype(np.float64, copy=False),
                    "plane_dir": plane_dir,
                }
            )
    print("method2: %d iscell ROIs pooled" % len(pool))
    return pool


def load_m1_mean_image() -> np.ndarray:
    imgs = []
    for v in range(1, 5):
        ops = load_npy(
            view_plane_dir(v, M1_DATA) / "ops.npy", allow_pickle=True
        ).item()
        imgs.append(np.asarray(ops["meanImg"], dtype=np.float64))
    return np.max(np.stack(imgs, axis=0), axis=0)


def load_m2_ref_mean_image() -> np.ndarray:
    # Prefer a mid plane (Plane 19 on this dataset); fall back to first in PLANE_SPECS
    mid = [p for p in PLANE_SPECS if p[1] == MATCH_MID_PLANE]
    plane_dir = mid[0][0] if mid else PLANE_SPECS[0][0]
    ops = load_npy(plane_dir / "ops.npy", allow_pickle=True).item()
    return np.asarray(ops["meanImg"], dtype=np.float64)


# ---------------------------------------------------------------------------
# XY registration: x2 = s*col + tx, y2 = s*row + ty
# ---------------------------------------------------------------------------
def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return -1.0
    return float(np.dot(a, b) / denom)


def estimate_xy_transform(
    img1: np.ndarray,
    img2: np.ndarray,
) -> dict[str, float]:
    """Estimate uniform scale + translation mapping method1 coords -> method2.

    Strategy: for candidate scales, resize img1, pad/crop to img2 shape with
    FFT phase-correlation for translation; pick max NCC.
    """
    from scipy.signal import fftconvolve

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    ref = img2.astype(np.float64)
    ref = (ref - ref.mean()) / (ref.std() + 1e-12)

    # Candidate scales: FOV-ratio, 1.0 (center-crop-ish), and nearby
    s0 = float(w2) / float(w1)
    scales = sorted(
        set(
            np.round(
                np.concatenate(
                    [
                        [s0],
                        np.linspace(s0 * 0.85, s0 * 1.15, 9),
                        [1.0 * w2 / w1],  # same as s0
                        np.linspace(0.70, 0.95, 6),
                    ]
                ),
                4,
            )
        )
    )
    # Also try exact center-crop equivalent: s=1, tx=ty=(406-512)/2 if we
    # conceptually crop — covered by trying s near 1 with translation.

    best = {"s": s0, "tx": (w2 - s0 * w1) / 2.0, "ty": (h2 - s0 * h1) / 2.0, "ncc": -2.0}

    for s in scales:
        scaled = ndi_zoom(img1, s, order=1)
        hs, ws = scaled.shape
        # Place scaled image into a canvas and correlate with ref via FFT
        # Build padded version for phase correlation against ref
        canvas = np.zeros_like(ref)
        # Initial placement: center
        y0 = (h2 - hs) // 2
        x0 = (w2 - ws) // 2
        # If scaled larger than ref, center-crop
        if hs >= h2 and ws >= w2:
            cy = (hs - h2) // 2
            cx = (ws - w2) // 2
            patch = scaled[cy : cy + h2, cx : cx + w2]
            # translation in method1->method2: x2 = s*x1 + tx
            # with center crop: x2 = s*x1 - s*cx_offset_in_scaled... 
            # When we crop scaled[cy:cy+h2, cx:cx+w2], pixel (x2,y2) in patch
            # corresponds to scaled (x2+cx, y2+cy) = (s*x1, s*y1) approx
            # so x2 = s*x1 - cx, y2 = s*y1 - cy
            mov = (patch - patch.mean()) / (patch.std() + 1e-12)
            # Fine translation via phase correlation
            corr = fftconvolve(ref, mov[::-1, ::-1], mode="same")
            peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
            dy = int(peak[0] - h2 // 2)
            dx = int(peak[1] - w2 // 2)
            # Apply shift to mov
            shifted = np.roll(np.roll(mov, dy, axis=0), dx, axis=1)
            ncc = _ncc(ref, shifted)
            tx = -float(cx) + dx
            ty = -float(cy) + dy
        else:
            # Pad scaled into canvas then phase-correlate
            y0 = max(0, (h2 - hs) // 2)
            x0 = max(0, (w2 - ws) // 2)
            y1 = min(h2, y0 + hs)
            x1 = min(w2, x0 + ws)
            sy0 = 0 if y0 >= 0 else -y0
            sx0 = 0 if x0 >= 0 else -x0
            canvas[:, :] = 0
            canvas[y0:y1, x0:x1] = scaled[
                sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)
            ]
            mov = (canvas - canvas.mean()) / (canvas.std() + 1e-12)
            corr = fftconvolve(ref, mov[::-1, ::-1], mode="same")
            peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
            dy = int(peak[0] - h2 // 2)
            dx = int(peak[1] - w2 // 2)
            shifted = np.roll(np.roll(mov, dy, axis=0), dx, axis=1)
            ncc = _ncc(ref, shifted)
            # canvas pixel (x0+sx, y0+sy) = scaled (sx,sy) ≈ (s*x1, s*y1)
            # after roll by (dx,dy): content moves so new[i]=old[i-shift]
            # Effective: x2 = s*x1 + x0 + dx  (approx)
            tx = float(x0 + dx)
            ty = float(y0 + dy)

        if ncc > best["ncc"]:
            best = {"s": float(s), "tx": float(tx), "ty": float(ty), "ncc": float(ncc)}

    # Fallback if NCC terrible: scale FOV ratio + center align
    if best["ncc"] < 0.05:
        s = s0
        best = {
            "s": s,
            "tx": (w2 - s * w1) / 2.0,
            "ty": (h2 - s * h1) / 2.0,
            "ncc": best["ncc"],
            "fallback": 1.0,
        }
        print("XY reg: low NCC=%.3f — fallback s=%.4f center-align" % (best["ncc"], s))
    else:
        print(
            "XY reg: s=%.4f  tx=%.2f  ty=%.2f  NCC=%.3f"
            % (best["s"], best["tx"], best["ty"], best["ncc"])
        )
    return best


def apply_xy(col: np.ndarray, row: np.ndarray, tfm: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    s, tx, ty = float(tfm["s"]), float(tfm["tx"]), float(tfm["ty"])
    return s * col + tx, s * row + ty


def plot_xy_overlay(
    img1: np.ndarray,
    img2: np.ndarray,
    tfm: dict[str, float],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    s, tx, ty = float(tfm["s"]), float(tfm["tx"]), float(tfm["ty"])
    h2, w2 = img2.shape
    yy, xx = np.mgrid[0:h2, 0:w2]
    # inverse: col = (x2 - tx)/s
    src_x = (xx - tx) / s
    src_y = (yy - ty) / s
    # bilinear sample img1
    from scipy.ndimage import map_coordinates

    warped = map_coordinates(
        img1.astype(np.float64),
        [src_y.ravel(), src_x.ravel()],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(h2, w2)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img1, cmap="gray")
    ax[0].set_title("method1 mean (Views)")
    ax[1].imshow(img2, cmap="gray")
    ax[1].set_title("method2 Plane%d mean" % MATCH_MID_PLANE)
    # color overlay
    a = warped - warped.min()
    a = a / (a.max() + 1e-12)
    b = img2.astype(np.float64)
    b = b - b.min()
    b = b / (b.max() + 1e-12)
    rgb = np.zeros((h2, w2, 3), dtype=np.float64)
    rgb[:, :, 0] = a
    rgb[:, :, 1] = b
    rgb[:, :, 2] = 0.3 * a + 0.3 * b
    ax[2].imshow(np.clip(rgb, 0, 1))
    ax[2].set_title("overlay R=m1warp G=m2  NCC~%.3f" % float(tfm.get("ncc", np.nan)))
    for a_ in ax:
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print("wrote", out_path)


# ---------------------------------------------------------------------------
# Method1 primary-view traces (full T)
# ---------------------------------------------------------------------------
def extract_m1_primary_traces(
    entries: list[dict[str, Any]],
    primaries: np.ndarray,
) -> np.ndarray:
    """(N, T) mean fluorescence at primary-view footprint from data.bin."""
    # open 4 memmaps
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

    from tplfm_utils import compute_move as _compute_move

    coef = np.load(str(M1_OUT / "psf_fit_coef.npz"))
    centers_1b = np.asarray(coef["centers_1based"], dtype=np.float64)
    move = _compute_move(centers_1b, factor=MOVE_DIVISOR)
    move_i = np.round(move).astype(int)

    n = len(entries)
    out = np.zeros((n, nf), dtype=np.float64)
    # Group by primary view for one-pass extraction
    by_view: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
    for i, pv in enumerate(primaries):
        by_view[int(pv)].append(i)

    frame_chunk = 100
    for pv, idxs in by_view.items():
        if not idxs:
            continue
        v0 = pv - 1
        m0, m1 = int(move_i[v0, 0]), int(move_i[v0, 1])
        raw = mems[v0]
        pix = []
        for i in idxs:
            e = entries[i]
            cav = np.asarray(e["center_allview"], dtype=np.float64)
            if cav.shape != (2, 4):
                cav = cav.reshape(2, 4)
            d1 = np.asarray(e["pixels1"], dtype=np.float64).ravel()
            d2 = np.asarray(e["pixels2"], dtype=np.float64).ravel()
            cx, cy = float(cav[0, v0]), float(cav[1, v0])
            px = np.rint(d1 + cx).astype(int)
            py = np.rint(d2 + cy).astype(int)
            ox = np.clip(px + m1, 0, lx - 1)
            oy = np.clip(py + m0, 0, ly - 1)
            pix.append((oy, ox))
        print("  extract m1 primary view%d: %d neurons, T=%d" % (pv, len(idxs), nf))
        for s in range(0, nf, frame_chunk):
            e = min(s + frame_chunk, nf)
            block = np.asarray(raw[s:e], dtype=np.float32)
            for k, i in enumerate(idxs):
                oy, ox = pix[k]
                if oy.size == 0:
                    continue
                out[i, s:e] = block[:, oy, ox].mean(axis=1)
    return out


def zscore(tr: np.ndarray) -> np.ndarray:
    tr = np.asarray(tr, dtype=np.float64).ravel()
    s = float(tr.std())
    if s < 1e-12:
        return np.zeros_like(tr)
    return (tr - tr.mean()) / s


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def build_cost_and_match(
    c1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    z2: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return row_ind, col_ind, dxy, dz, corr for assigned pairs (gated)."""
    n1, n2 = c1.shape[0], x2.shape[0]
    cost = np.full((n1, n2), BIG, dtype=np.float64)
    dxy_m = np.full((n1, n2), np.nan)
    dz_m = np.full((n1, n2), np.nan)
    r_m = np.full((n1, n2), np.nan)

    # Pre-zscore traces
    t1z = np.stack([zscore(t1[i]) for i in range(n1)], axis=0)
    t2z = np.stack([zscore(t2[j]) for j in range(n2)], axis=0)

    for i in range(n1):
        x1i, y1i, z1i = c1[i, 0], c1[i, 1], c1[i, 2]
        for j in range(n2):
            dxy = float(np.hypot(x1i - x2[j], y1i - y2[j]))
            dz = float(abs(z1i - z2[j]))
            if dxy > D_XY_PX or dz > D_Z_UM:
                continue
            # Pearson on z-scored = dot / (n-1) roughly; use corrcoef_scalar
            r = corrcoef_scalar(t1z[i], t2z[j])
            if not np.isfinite(r) or r < R_MIN:
                continue
            dxy_m[i, j] = dxy
            dz_m[i, j] = dz
            r_m[i, j] = r
            cost[i, j] = W_XY * dxy + W_Z * dz + W_R * (1.0 - r)

    r_ind, c_ind = linear_sum_assignment(cost)
    # Keep only finite (non-gated) assignments
    keep = cost[r_ind, c_ind] < BIG * 0.5
    return (
        r_ind[keep],
        c_ind[keep],
        dxy_m[r_ind[keep], c_ind[keep]],
        dz_m[r_ind[keep], c_ind[keep]],
        r_m[r_ind[keep], c_ind[keep]],
    )


def load_matches_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = []
    for r in rows:
        out.append(
            {
                "m1_idx": int(r["m1_idx"]),
                "m2_plane": int(r["m2_plane"]),
                "m2_roi": int(r["m2_roi"]),
                "col1": float(r["col1"]),
                "row1": float(r["row1"]),
                "z1": float(r["z1"]),
                "x2": float(r["x2"]),
                "y2": float(r["y2"]),
                "z2": float(r["z2"]),
                "x1_in_m2": float(r["x1_in_m2"]),
                "y1_in_m2": float(r["y1_in_m2"]),
                "d_xy_px": float(r["d_xy_px"]),
                "d_z_um": float(r["d_z_um"]),
                "corr": float(r["corr"]),
                "cost": float(r.get("cost", np.nan)),
            }
        )
    return out


def _m2_index_map(m2_pool: list[dict[str, Any]]) -> dict[tuple[int, int], int]:
    return {(m["plane_id"], m["roi_idx"]): i for i, m in enumerate(m2_pool)}


def _m1_primary_pixels(
    entry: dict[str, Any],
    primary: int,
    ly: int,
    lx: int,
    move_i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Absolute (oy, ox) footprint in rolled FOV + center (cx, cy) pre-roll."""
    v0 = int(primary) - 1
    cav = np.asarray(entry["center_allview"], dtype=np.float64)
    if cav.shape != (2, 4):
        cav = cav.reshape(2, 4)
    d1 = np.asarray(entry["pixels1"], dtype=np.float64).ravel()
    d2 = np.asarray(entry["pixels2"], dtype=np.float64).ravel()
    cx, cy = float(cav[0, v0]), float(cav[1, v0])
    px = np.rint(d1 + cx).astype(int)
    py = np.rint(d2 + cy).astype(int)
    m0, m1 = int(move_i[v0, 0]), int(move_i[v0, 1])
    ox = np.clip(px + m1, 0, lx - 1)
    oy = np.clip(py + m0, 0, ly - 1)
    return oy, ox, float(cx + m1), float(cy + m0)


def _crop_extent(
    xs: np.ndarray,
    ys: np.ndarray,
    h: int,
    w: int,
    pad: int = 40,
) -> tuple[int, int, int, int]:
    if xs.size == 0:
        return 0, w, 0, h
    x0 = int(max(0, np.floor(xs.min()) - pad))
    x1 = int(min(w, np.ceil(xs.max()) + pad + 1))
    y0 = int(max(0, np.floor(ys.min()) - pad))
    y1 = int(min(h, np.ceil(ys.max()) + pad + 1))
    if x1 <= x0:
        x0, x1 = 0, w
    if y1 <= y0:
        y0, y1 = 0, h
    return x0, x1, y0, y1


def plot_matched_pairs(
    matches: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    primaries: np.ndarray,
    m2_pool: list[dict[str, Any]],
    t1: np.ndarray,
    out_dir: Path,
) -> None:
    """One PNG per matched pair: footprints + overlaid activity."""
    import matplotlib.pyplot as plt

    from tplfm_utils import compute_move as _compute_move

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("match_*.png"):
        old.unlink()

    # method1 mean images per view (rolled FOV as used for data.bin)
    m1_means: list[np.ndarray] = []
    ly = lx = None
    for v in range(1, 5):
        ops = load_npy(
            view_plane_dir(v, M1_DATA) / "ops.npy", allow_pickle=True
        ).item()
        img = np.asarray(ops["meanImg"], dtype=np.float64)
        m1_means.append(img)
        ly, lx = int(ops["Ly"]), int(ops["Lx"])
    assert ly is not None and lx is not None

    coef = np.load(str(M1_OUT / "psf_fit_coef.npz"))
    move_i = np.round(
        _compute_move(
            np.asarray(coef["centers_1based"], dtype=np.float64), factor=MOVE_DIVISOR
        )
    ).astype(int)
    # meanImg / data.bin are unrolled; footprints use +move (same as extract_m1_primary_traces)

    m2_means: dict[int, np.ndarray] = {}
    for plane_dir, plane_id, _z in PLANE_SPECS:
        ops = load_npy(plane_dir / "ops.npy", allow_pickle=True).item()
        m2_means[int(plane_id)] = np.asarray(ops["meanImg"], dtype=np.float64)

    m2_map = _m2_index_map(m2_pool)
    n_t = t1.shape[1]
    t = np.arange(n_t)

    n_ok = 0
    for k, row in enumerate(matches):
        i1 = int(row["m1_idx"])
        key = (int(row["m2_plane"]), int(row["m2_roi"]))
        if key not in m2_map:
            print("  skip match %d: m2 plane%d roi%d not in pool" % (k, key[0], key[1]))
            continue
        j2 = m2_map[key]
        m2 = m2_pool[j2]
        entry = entries[i1]
        pv = int(primaries[i1])
        oy, ox, cx_r, cy_r = _m1_primary_pixels(entry, pv, ly, lx, move_i)
        img1 = m1_means[pv - 1]
        img2 = m2_means[int(m2["plane_id"])]
        x2p = np.asarray(m2["xpix"], dtype=np.float64)
        y2p = np.asarray(m2["ypix"], dtype=np.float64)

        tr1 = np.asarray(t1[i1, :n_t], dtype=np.float64)
        tr2 = np.asarray(m2["trace"][:n_t], dtype=np.float64)
        z1 = zscore(tr1)
        z2 = zscore(tr2)
        r = float(row["corr"])

        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(
            3, 2, height_ratios=[1.35, 1.0, 1.0], hspace=0.38, wspace=0.28
        )
        ax_m1 = fig.add_subplot(gs[0, 0])
        ax_m2 = fig.add_subplot(gs[0, 1])
        ax_z = fig.add_subplot(gs[1, :])
        ax_r1 = fig.add_subplot(gs[2, 0])
        ax_r2 = fig.add_subplot(gs[2, 1], sharex=ax_r1)

        # --- m1 footprint crop ---
        x0, x1, y0, y1 = _crop_extent(ox.astype(float), oy.astype(float), ly, lx, pad=45)
        crop1 = img1[y0:y1, x0:x1]
        ax_m1.imshow(
            crop1,
            cmap="gray",
            origin="upper",
            extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5),
            vmin=np.percentile(crop1, 2) if crop1.size else 0,
            vmax=np.percentile(crop1, 98) if crop1.size else 1,
        )
        if ox.size:
            ax_m1.scatter(ox, oy, s=6, c="C3", alpha=0.35, linewidths=0)
        ax_m1.plot(cx_r, cy_r, "x", color="C3", ms=10, mew=2)
        ax_m1.set_aspect("equal")
        ax_m1.set_title(
            "m1 idx=%d  primary=v%d  z=%.1f um" % (i1, pv, float(row["z1"])),
            fontsize=10,
        )
        ax_m1.set_xlabel("col")
        ax_m1.set_ylabel("row")

        # --- m2 footprint crop ---
        h2, w2 = img2.shape
        x0b, x1b, y0b, y1b = _crop_extent(x2p, y2p, h2, w2, pad=45)
        crop2 = img2[y0b:y1b, x0b:x1b]
        ax_m2.imshow(
            crop2,
            cmap="gray",
            origin="upper",
            extent=(x0b - 0.5, x1b - 0.5, y1b - 0.5, y0b - 0.5),
            vmin=np.percentile(crop2, 2) if crop2.size else 0,
            vmax=np.percentile(crop2, 98) if crop2.size else 1,
        )
        if x2p.size:
            ax_m2.scatter(x2p, y2p, s=6, c="C0", alpha=0.45, linewidths=0)
        ax_m2.plot(float(m2["x"]), float(m2["y"]), "x", color="C0", ms=10, mew=2)
        ax_m2.plot(
            float(row["x1_in_m2"]),
            float(row["y1_in_m2"]),
            "+",
            color="C3",
            ms=11,
            mew=1.8,
            label="m1→m2",
        )
        ax_m2.legend(loc="upper right", fontsize=7)
        ax_m2.set_aspect("equal")
        ax_m2.set_title(
            "m2 plane%d roi%d  z=%.1f um" % (m2["plane_id"], m2["roi_idx"], float(row["z2"])),
            fontsize=10,
        )
        ax_m2.set_xlabel("x")
        ax_m2.set_ylabel("y")

        # --- z-scored overlay ---
        ax_z.plot(t, z1, color="C3", lw=0.55, label="m1 primary", alpha=0.9)
        ax_z.plot(t, z2, color="C0", lw=0.55, label="m2 F", alpha=0.85)
        ax_z.set_ylabel("z-score")
        ax_z.grid(True, alpha=0.3)
        ax_z.legend(loc="upper right", fontsize=8, ncol=2)
        ax_z.set_title(
            "activity  corr=%.3f  d_xy=%.1f px  |dz|=%.1f um"
            % (r, float(row["d_xy_px"]), float(row["d_z_um"])),
            fontsize=10,
        )

        ax_r1.plot(t, tr1, color="C3", lw=0.45)
        ax_r1.set_title("m1 raw F", fontsize=9)
        ax_r1.set_ylabel("F")
        ax_r1.grid(True, alpha=0.3)
        ax_r1.set_xlabel("frame")

        ax_r2.plot(t, tr2, color="C0", lw=0.45)
        ax_r2.set_title("m2 raw F", fontsize=9)
        ax_r2.set_ylabel("F")
        ax_r2.grid(True, alpha=0.3)
        ax_r2.set_xlabel("frame")

        fig.suptitle(
            "match %03d | m1=%d ↔ plane%d/roi%d | r=%.3f"
            % (k, i1, m2["plane_id"], m2["roi_idx"], r),
            fontsize=12,
            y=0.995,
        )
        fig.subplots_adjust(top=0.93, bottom=0.06, left=0.07, right=0.98)
        path = out_dir / (
            "match_%03d_m1-%03d_p%d_roi%03d.png"
            % (k, i1, m2["plane_id"], m2["roi_idx"])
        )
        fig.savefig(str(path), dpi=100)
        plt.close(fig)
        n_ok += 1
        if (k + 1) % 25 == 0 or k == 0 or k + 1 == len(matches):
            print("  wrote %d/%d -> %s" % (k + 1, len(matches), path.name))

    print("wrote %d match pair figures in %s" % (n_ok, out_dir))


def export_results(
    out_dir: Path,
    m1_centers_orig: np.ndarray,
    m1_centers_m2: np.ndarray,
    m2_pool: list[dict[str, Any]],
    r_ind: np.ndarray,
    c_ind: np.ndarray,
    dxy: np.ndarray,
    dz: np.ndarray,
    corr: np.ndarray,
    cost_vals: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    match_rows = []
    matched_m1 = set(int(i) for i in r_ind)
    matched_m2 = set(int(j) for j in c_ind)

    for k, (i, j) in enumerate(zip(r_ind, c_ind)):
        m2 = m2_pool[int(j)]
        match_rows.append(
            {
                "m1_idx": int(i),
                "m2_plane": m2["plane_id"],
                "m2_roi": m2["roi_idx"],
                "col1": m1_centers_orig[i, 0],
                "row1": m1_centers_orig[i, 1],
                "z1": m1_centers_orig[i, 2],
                "x2": m2["x"],
                "y2": m2["y"],
                "z2": m2["z_um"],
                "x1_in_m2": m1_centers_m2[i, 0],
                "y1_in_m2": m1_centers_m2[i, 1],
                "d_xy_px": float(dxy[k]),
                "d_z_um": float(dz[k]),
                "corr": float(corr[k]),
                "cost": float(cost_vals[k]),
            }
        )

    with open(out_dir / "matches.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(match_rows[0].keys()) if match_rows else [
            "m1_idx", "m2_plane", "m2_roi"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(match_rows)
    print("wrote", out_dir / "matches.csv", "n=%d" % len(match_rows))

    um1 = []
    for i in range(m1_centers_orig.shape[0]):
        if i not in matched_m1:
            um1.append(
                {
                    "m1_idx": i,
                    "col1": m1_centers_orig[i, 0],
                    "row1": m1_centers_orig[i, 1],
                    "z1": m1_centers_orig[i, 2],
                }
            )
    with open(out_dir / "unmatched_m1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["m1_idx", "col1", "row1", "z1"])
        w.writeheader()
        w.writerows(um1)

    um2 = []
    for j, m2 in enumerate(m2_pool):
        if j not in matched_m2:
            um2.append(
                {
                    "m2_plane": m2["plane_id"],
                    "m2_roi": m2["roi_idx"],
                    "x2": m2["x"],
                    "y2": m2["y"],
                    "z2": m2["z_um"],
                }
            )
    with open(out_dir / "unmatched_m2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["m2_plane", "m2_roi", "x2", "y2", "z2"]
        )
        w.writeheader()
        w.writerows(um2)
    print("unmatched m1=%d  m2=%d" % (len(um1), len(um2)))

    # scatter with links
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        [m["x"] for m in m2_pool],
        [m["y"] for m in m2_pool],
        c="0.7",
        s=18,
        label="m2 all",
        zorder=1,
    )
    ax.scatter(
        m1_centers_m2[:, 0],
        m1_centers_m2[:, 1],
        c=m1_centers_orig[:, 2],
        cmap="viridis",
        s=28,
        marker="x",
        label="m1 (warped)",
        zorder=2,
    )
    for k, (i, j) in enumerate(zip(r_ind, c_ind)):
        m2 = m2_pool[int(j)]
        ax.plot(
            [m1_centers_m2[i, 0], m2["x"]],
            [m1_centers_m2[i, 1], m2["y"]],
            "C3-",
            lw=0.7,
            alpha=0.7,
            zorder=3,
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (method2 px)")
    ax.set_ylabel("y (method2 px)")
    ax.set_title("matches n=%d" % len(match_rows))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_dir / "match_xy_scatter.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(corr):
        ax.hist(corr, bins=min(30, max(8, len(corr) // 3)), color="C0", edgecolor="k")
        ax.axvline(float(np.median(corr)), color="C3", ls="--", label="median=%.3f" % float(np.median(corr)))
        ax.legend()
    ax.set_xlabel("trace corr")
    ax.set_ylabel("count")
    ax.set_title("match correlation")
    fig.tight_layout()
    fig.savefig(str(out_dir / "match_corr_hist.png"), dpi=120)
    plt.close(fig)
    print("wrote match_xy_scatter.png, match_corr_hist.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("OUT_DIR =", OUT_DIR)
    print(
        "MATCH_MODE=%s  plane z: %s"
        % (MATCH_MODE, {pid: z for _p, pid, z in PLANE_SPECS})
    )

    if PLOT_MATCHES_ONLY:
        print("mode: PLOT_MATCHES_ONLY (use existing matches.csv)")
        matches = load_matches_csv(OUT_DIR / "matches.csv")
        print("matches: %d" % len(matches))
        print("\n=== load method1 / method2 ===")
        entries, _c1_orig, primaries = load_method1()
        m2_pool = load_method2()
        print("\n=== extract method1 primary-view traces ===")
        t1 = extract_m1_primary_traces(entries, primaries)
        n_t = min(t1.shape[1], int(m2_pool[0]["trace"].shape[0]))
        t1 = t1[:, :n_t]
        print("\n=== plot matched pairs ===")
        plot_matched_pairs(
            matches, entries, primaries, m2_pool, t1, MATCH_PLOT_DIR
        )
        print("\nDone.")
        return

    print(
        "gates: D_XY=%.1fpx  D_Z=%.1fum  R_MIN=%.2f  W=(%.1f,%.1f,%.1f)"
        % (D_XY_PX, D_Z_UM, R_MIN, W_XY, W_Z, W_R)
    )

    print("\n=== load method1 / method2 ===")
    entries, c1_orig, primaries = load_method1()
    m2_pool = load_method2()

    print("\n=== XY registration ===")
    img1 = load_m1_mean_image()
    img2 = load_m2_ref_mean_image()
    tfm = estimate_xy_transform(img1, img2)
    np.savez(
        str(OUT_DIR / "xy_transform.npz"),
        s=tfm["s"],
        tx=tfm["tx"],
        ty=tfm["ty"],
        ncc=tfm.get("ncc", np.nan),
    )
    plot_xy_overlay(img1, img2, tfm, OUT_DIR / "xy_overlay.png")

    x1m2, y1m2 = apply_xy(c1_orig[:, 0], c1_orig[:, 1], tfm)
    c1_m2 = np.column_stack([x1m2, y1m2, c1_orig[:, 2]])

    print("\n=== extract method1 primary-view traces ===")
    t1 = extract_m1_primary_traces(entries, primaries)
    t2 = np.stack([m["trace"] for m in m2_pool], axis=0)
    # align lengths
    n_t = min(t1.shape[1], t2.shape[1])
    t1 = t1[:, :n_t]
    t2 = t2[:, :n_t]
    step = max(int(TRACE_T_STRIDE), 1)
    if step > 1:
        t1 = t1[:, ::step]
        t2 = t2[:, ::step]
    print(
        "traces T=%d  stride=%d  m1=%s  m2=%s"
        % (t1.shape[1], step, t1.shape, t2.shape)
    )

    print("\n=== Hungarian match ===")
    x2 = np.array([m["x"] for m in m2_pool], dtype=np.float64)
    y2 = np.array([m["y"] for m in m2_pool], dtype=np.float64)
    z2 = np.array([m["z_um"] for m in m2_pool], dtype=np.float64)
    r_ind, c_ind, dxy, dz, corr = build_cost_and_match(
        c1_m2, x2, y2, z2, t1, t2
    )
    cost_vals = W_XY * dxy + W_Z * dz + W_R * (1.0 - corr)

    # per-plane hits
    plane_hits: dict[int, int] = {}
    for j in c_ind:
        pid = int(m2_pool[int(j)]["plane_id"])
        plane_hits[pid] = plane_hits.get(pid, 0) + 1

    print(
        "matched %d / m1=%d / m2=%d"
        % (len(r_ind), c1_orig.shape[0], len(m2_pool))
    )
    if len(corr):
        print(
            "median d_xy=%.2f px  median |dz|=%.2f um  median r=%.3f"
            % (float(np.median(dxy)), float(np.median(dz)), float(np.median(corr)))
        )
    print("hits by plane:", plane_hits)

    export_results(
        OUT_DIR,
        c1_orig,
        c1_m2,
        m2_pool,
        r_ind,
        c_ind,
        dxy,
        dz,
        corr,
        cost_vals,
    )

    if SKIP_PAIR_PLOTS:
        print("SKIP_PAIR_PLOTS=True — skip per-pair PNGs")
    else:
        print("\n=== plot matched pairs ===")
        matches = load_matches_csv(OUT_DIR / "matches.csv")
        plot_matched_pairs(matches, entries, primaries, m2_pool, t1, MATCH_PLOT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
