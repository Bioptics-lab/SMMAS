# -*- coding: utf-8 -*-
"""Shared helpers for TPLFM localization pipeline (MATLAB ports)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CODE_DIR.parent  # package root (PSF/, Views/, outputs)

# Default mid-depth PSF centers (1-based row, col), same as MATLAB comments
VIEW_CENTERS_1BASED = np.array(
    [[164, 145], [136, 97], [119, 130], [153, 174]], dtype=np.float64
)

CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def ensure_numpy2_pickle_compat() -> None:
    """Allow loading suite2p .npy pickled with NumPy 2.x under NumPy 1.x."""
    if "numpy._core" in sys.modules:
        return
    import numpy.core as nc

    sys.modules["numpy._core"] = nc
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric
    try:
        sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath
    except Exception:
        pass


def load_npy(path: Path | str, allow_pickle: bool = True):
    ensure_numpy2_pickle_compat()
    return np.load(str(path), allow_pickle=allow_pickle)


def compute_move(
    centers: np.ndarray | None = None,
    factor: float = 5.0,
) -> np.ndarray:
    """Inter-view alignment offsets from PSF centers (MATLAB NeuronPlot / estimate_depth)."""
    if centers is None:
        centers = VIEW_CENTERS_1BASED.copy()
    c = np.asarray(centers, dtype=np.float64)
    c = c - c.mean(axis=0, keepdims=True)
    return np.round(c / factor).astype(np.float64)


def corrcoef_scalar(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r; NaN if undefined (matches corrcoef(1,2) usage)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def dxdy_psf(
    depth: float,
    view_idx: int,
    coef_psf_xz: np.ndarray,
    coef_psf_yz: np.ndarray,
    move: np.ndarray | None = None,
    factor: float = 5.0,
    x0: float = 60.0,
    y0: float = 60.0,
) -> tuple[float, float]:
    """Parallax offset at depth (independent x(z), y(z)). view_idx 0-based."""
    dx, dy = parallax_dx_dy(
        depth, view_idx, coef_psf_xz, coef_psf_yz, factor=factor, x0=x0, y0=y0
    )
    if move is not None:
        dx = -dx + float(move[view_idx, 0])
        dy = -dy + float(move[view_idx, 1])
    return float(dx), float(dy)


def parallax_dx_dy(
    nz: float,
    view_idx: int,
    coef_psf_xz: np.ndarray,
    coef_psf_yz: np.ndarray,
    factor: float = 5.0,
    x0: float = 60.0,
    y0: float = 60.0,
) -> tuple[float, float]:
    """Lateral PSF peak offset vs mid-crop at relative depth nz (plane units).

    Uses independent x(z) and y(z) fits. (Old MATLAB chained y(x) breaks when
    a view shears only in Y.)
    """
    dx = (
        coef_psf_xz[0, view_idx] / factor * nz
        + coef_psf_xz[1, view_idx] / factor
        - x0 / factor
    )
    dy = (
        coef_psf_yz[0, view_idx] / factor * nz
        + coef_psf_yz[1, view_idx] / factor
        - y0 / factor
    )
    return float(dx), float(dy)
