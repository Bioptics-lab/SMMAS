# -*- coding: utf-8 -*-
"""Fit PSF centroid trajectories vs depth (extended from psf_fit.m).

Original MATLAB fitted x(z) then y(x). That fails for views whose PSF
shears mainly in Y (views 1–2 here). We fit independent x(z) and y(z).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from tplfm_utils import SCRIPT_DIR


def _largest_blob_centroid(image: np.ndarray) -> tuple[float, float]:
    """Return (x/col, y/row) centroid; 1-based like MATLAB regionprops."""
    labeled, n_lab = ndimage.label(image > 0)
    if n_lab == 0:
        cy, cx = (np.array(image.shape) - 1) / 2.0
        return float(cx + 1.0), float(cy + 1.0)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    ind = int(np.argmax(counts))
    ys, xs = np.nonzero(labeled == ind)
    return float(xs.mean() + 1.0), float(ys.mean() + 1.0)


def _intensity_centroid(image: np.ndarray, thr_frac: float = 0.05) -> tuple[float, float]:
    """Intensity-weighted (x/col, y/row); 1-based. Prefer for unthresholded crops."""
    img = np.asarray(image, dtype=np.float64)
    peak = float(np.max(img)) if img.size else 0.0
    if peak <= 0:
        cy, cx = (np.array(img.shape) - 1) / 2.0
        return float(cx + 1.0), float(cy + 1.0)
    mask = img > (peak * thr_frac)
    if not np.any(mask):
        mask = img > 0
    ys, xs = np.nonzero(mask)
    w = img[ys, xs]
    wsum = float(w.sum())
    if wsum <= 0:
        return float(xs.mean() + 1.0), float(ys.mean() + 1.0)
    return float(np.sum(xs * w) / wsum + 1.0), float(np.sum(ys * w) / wsum + 1.0)


def psf_fit(
    psf_parts: np.ndarray,
    z_um: np.ndarray | None = None,
    *,
    centroid: str = "binary",
    z_abs_max: float | None = None,
    border_margin_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Parameters
    ----------
    psf_parts : (Y, X, Z, view)
    z_um : depth axis; default 1 unit per plane, centered (matches plane-index depth)
    centroid : \"binary\" = MATLAB-style support mean after label>0 (needs thresholded
        PSFs); \"intensity\" = brightness-weighted (needed for raw/sim crops where
        zoom leaves tiny positives everywhere so binary support fills the frame).
    z_abs_max : if set, only fit planes with |z_um| <= z_abs_max (avoids crop-edge
        clipping that flattens global slopes).
    border_margin_px : if set, drop planes whose intensity peak is within this many
        pixels of the crop border (auto unclipped window; overrides nothing if
        z_abs_max already set — both filters AND together).

    Returns
    -------
    fit_line : (2, Z, view) [x/col; y/row] 1-based
    coef_psf_xz : (2, view) polyfit(z -> x)
    coef_psf_yz : (2, view) polyfit(z -> y)  **use this for parallax**
    coef_psf_xy : (2, view) polyfit(x -> y)  (MATLAB legacy)
    x0, y0 : crop-center references (1-based) for subtracting mid
    """
    psf_parts = np.asarray(psf_parts, dtype=np.float64)
    if psf_parts.ndim != 4:
        raise ValueError("psf_parts must be (Y,X,Z,view), got %s" % (psf_parts.shape,))
    n_y, n_x, n_z, n_views = psf_parts.shape
    if z_um is None:
        # Relative plane index centered on *this stack's* mid slice.
        # Prefer passing absolute z_rel = z_abs0 - (n_z_full-1)/2 so it matches
        # simulation GT (full 0..n_z_full-1), especially when the stack is a
        # cropped Z range (e.g. planes 6..115 of 121).
        z_um = np.arange(n_z, dtype=np.float64) - (n_z - 1) / 2.0
    z_um = np.asarray(z_um, dtype=np.float64).ravel()
    if z_um.size != n_z:
        raise ValueError("z_um length %d != n_z %d" % (z_um.size, n_z))

    x0 = (n_x + 1) / 2.0  # 1-based center col
    y0 = (n_y + 1) / 2.0  # 1-based center row

    fit_line = np.zeros((2, n_z, n_views), dtype=np.float64)
    for v in range(n_views):
        for iz in range(n_z):
            image = psf_parts[:, :, iz, v].copy()
            if centroid == "intensity":
                x, y = _intensity_centroid(image)
            else:
                labeled, n_lab = ndimage.label(image > 0)
                if n_lab > 0:
                    counts = np.bincount(labeled.ravel())
                    counts[0] = 0
                    ind = int(np.argmax(counts))
                    image[labeled != ind] = 0.0
                x, y = _largest_blob_centroid(image)
            fit_line[0, iz, v] = x
            fit_line[1, iz, v] = y

    # Planes used for polyfit (full stack still stored in fit_line)
    use = np.ones(n_z, dtype=bool)
    if z_abs_max is not None:
        use &= np.abs(z_um) <= float(z_abs_max)
    if border_margin_px is not None:
        m = float(border_margin_px)
        # 1-based peak must stay in (1+m, n_x-m) x (1+m, n_y-m)
        ok = np.ones(n_z, dtype=bool)
        for v in range(n_views):
            x = fit_line[0, :, v]
            y = fit_line[1, :, v]
            ok &= (x > 1.0 + m) & (x < n_x - m) & (y > 1.0 + m) & (y < n_y - m)
        use &= ok
    if int(use.sum()) < 5:
        raise ValueError(
            "too few planes for PSF polyfit after clipping filters (%d/%d)"
            % (int(use.sum()), n_z)
        )

    coef_psf_xz = np.zeros((2, n_views), dtype=np.float64)
    coef_psf_yz = np.zeros((2, n_views), dtype=np.float64)
    coef_psf_xy = np.zeros((2, n_views), dtype=np.float64)
    for v in range(n_views):
        coef_psf_xz[:, v] = np.polyfit(z_um[use], fit_line[0, use, v], 1)
        coef_psf_yz[:, v] = np.polyfit(z_um[use], fit_line[1, use, v], 1)
        coef_psf_xy[:, v] = np.polyfit(fit_line[0, use, v], fit_line[1, use, v], 1)

    return fit_line, coef_psf_xz, coef_psf_yz, coef_psf_xy, float(x0), float(y0)


def stack_views(psf_list: list[np.ndarray]) -> np.ndarray:
    return np.stack(psf_list, axis=-1)


def fit_from_energy_tiffs(psf_dir: Path | None = None):
    import tifffile

    psf_dir = Path(psf_dir) if psf_dir is not None else SCRIPT_DIR
    vols = []
    for v in range(1, 5):
        for name in ("psf%d_energy.tif" % v, "psf%d_seg.tif" % v):
            path = psf_dir / name
            if path.is_file():
                raw = tifffile.imread(str(path))
                if raw.shape[0] < raw.shape[1] and raw.shape[0] < raw.shape[2]:
                    raw = np.transpose(raw, (1, 2, 0))
                vols.append(np.asarray(raw, dtype=np.float64))
                break
        else:
            raise FileNotFoundError("missing segmented/energy PSF for view %d" % v)
    parts = stack_views(vols)
    fit_line, coef_xz, coef_yz, coef_xy, x0, y0 = psf_fit(parts)
    np.savez(
        str(psf_dir / "psf_fit_coef.npz"),
        fit_line=fit_line,
        coef_psf_xz=coef_xz,
        coef_psf_yz=coef_yz,
        coef_psf_xy=coef_xy,
        x0=x0,
        y0=y0,
    )
    print("saved psf_fit_coef.npz")
    print("  xz slopes", coef_xz[0])
    print("  yz slopes", coef_yz[0])
    return fit_line, coef_xz, coef_yz, coef_xy, x0, y0


if __name__ == "__main__":
    fit_from_energy_tiffs()
