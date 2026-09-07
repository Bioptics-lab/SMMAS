# -*- coding: utf-8 -*-
"""Segment / crop TPLFM PSFs (port of psf_seg_TPLFM.m).

Example (MATLAB comments, 1-based centers & z):
  psf_1n = psf_seg_TPLFM(psf1, [164, 145], 6:46)
  psf_2n = psf_seg_TPLFM(psf2, [136,  97], 6:46)
  psf_3n = psf_seg_TPLFM(psf3, [119, 130], 6:46)
  psf_4n = psf_seg_TPLFM(psf4, [153, 174], 6:46)

Four-view PSFs: PSF/file_1.tif .. file_4.tif at the package root (or pass psf_dir).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

CODE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CODE_DIR.parent  # package root; pass PSF_DIR from dataset_config in the pipeline

# MATLAB 1-based (row, col) at mid-depth; must cover PSF travel (~±106 px)
VIEW_CENTERS_1BASED = (
    (150, 150),
    (150, 150),
    (150, 150),
    (150, 150),
)
Z_START_1BASED = 6
Z_END_1BASED = 115  # inclusive
# Half-size must be > peak travel (~106) or centroids clip and fit fails
CROP_HALF_Y = 110  # -> 221 rows
CROP_HALF_X = 110  # -> 221 cols
THRESH_STD = 3.5
CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def load_psf_volume(path: Path) -> np.ndarray:
    """Load 3D TIFF as float64 (Y, X, Z)."""
    raw = tifffile.imread(str(path))
    if raw.ndim != 3:
        raise ValueError("PSF must be 3D, got %s from %s" % (raw.shape, path))
    # Prefer (Y,X,Z); if leading axis is small like Z stack from ImageJ, transpose
    if raw.shape[0] < raw.shape[1] and raw.shape[0] < raw.shape[2]:
        # Z,Y,X -> Y,X,Z
        vol = np.transpose(raw, (1, 2, 0))
    else:
        vol = raw
    return np.asarray(vol, dtype=np.float64)


def psf_seg_TPLFM(
    ch1: np.ndarray,
    center: tuple[int, int] | list[int] | np.ndarray,
    zrange: range | slice | np.ndarray | list[int],
    *,
    one_based: bool = True,
    crop_half_y: int | None = None,
    crop_half_x: int | None = None,
) -> np.ndarray:
    """Crop, threshold, and keep largest 4-connected blob per Z plane.

    Parameters
    ----------
    ch1 : (Y, X, Z) array
    center : (row, col); 1-based if one_based=True (MATLAB style)
    zrange : Z indices; 1-based inclusive if one_based=True (e.g. range(6, 47))
    one_based : match MATLAB indexing when True
    crop_half_y, crop_half_x : override default CROP_HALF_*; must fit in image

    Returns
    -------
    psf : (2*hy+1, 2*hx+1, n_z) float64
    """
    cy, cx = int(center[0]), int(center[1])
    if one_based:
        cy -= 1
        cx -= 1

    if isinstance(zrange, slice):
        z_indices = list(range(*zrange.indices(ch1.shape[2])))
    elif isinstance(zrange, range):
        z_indices = list(zrange)
    else:
        z_indices = [int(z) for z in np.asarray(zrange).ravel()]

    if one_based:
        z_indices = [z - 1 for z in z_indices]

    hy = int(CROP_HALF_Y if crop_half_y is None else crop_half_y)
    hx = int(CROP_HALF_X if crop_half_x is None else crop_half_x)
    n_y, n_x = int(ch1.shape[0]), int(ch1.shape[1])
    if cy - hy < 0 or cy + hy >= n_y or cx - hx < 0 or cx + hx >= n_x:
        raise ValueError(
            "crop around center (%d,%d) half=(%d,%d) exceeds image %dx%d"
            % (cy + (1 if one_based else 0), cx + (1 if one_based else 0), hy, hx, n_y, n_x)
        )

    n_z = len(z_indices)
    out = np.zeros((2 * hy + 1, 2 * hx + 1, n_z), dtype=np.float64)

    for k, iz in enumerate(z_indices):
        planei = np.asarray(ch1[:, :, iz], dtype=np.float64)
        stdd = float(planei.std(ddof=1))  # MATLAB std2 / std default (N-1)
        planei = planei - planei.mean()
        planei[planei < THRESH_STD * stdd] = 0.0

        y0, y1 = cy - hy, cy + hy + 1
        x0, x1 = cx - hx, cx + hx + 1
        planei = planei[y0:y1, x0:x1]

        labeled, n_lab = ndimage.label(planei > 0, structure=CONN4)
        if n_lab > 0:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            ind = int(np.argmax(counts))
            planei[labeled != ind] = 0.0

        out[:, :, k] = planei

    return out


def process_all_views(
    psf_dir: Path | None = None,
    z_start_1based: int = Z_START_1BASED,
    z_end_1based: int = Z_END_1BASED,
    save: bool = True,
) -> list[np.ndarray]:
    """Load psf1–4.tif, segment with default centers, optionally save."""
    psf_dir = Path(psf_dir) if psf_dir is not None else SCRIPT_DIR
    zrange = range(z_start_1based, z_end_1based + 1)
    results: list[np.ndarray] = []

    for v, center in enumerate(VIEW_CENTERS_1BASED, start=1):
        path = psf_dir / ("psf%d.tif" % v)
        print("Loading %s ..." % path.name)
        vol = load_psf_volume(path)
        print("  shape (Y,X,Z)=%s, center(1-based)=%s, z=%d:%d" % (
            vol.shape, center, z_start_1based, z_end_1based
        ))
        seg = psf_seg_TPLFM(vol, center, zrange, one_based=True)
        results.append(seg)
        if save:
            out_path = psf_dir / ("psf%d_seg.tif" % v)
            # save as Z,Y,X for ImageJ-friendly stacks
            tifffile.imwrite(str(out_path), np.transpose(seg, (2, 0, 1)))
            print("  saved %s  shape=%s" % (out_path.name, seg.shape))

    return results


if __name__ == "__main__":
    process_all_views()
