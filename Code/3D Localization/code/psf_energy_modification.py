# -*- coding: utf-8 -*-
"""Compensate Z-dependent PSF energy decay (port of psf_energy_modification.m)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from psf_seg_TPLFM import SCRIPT_DIR, process_all_views


def xy_sum_curve(psf: np.ndarray) -> np.ndarray:
    """Sum over Y,X -> (n_z,) intensity vs depth."""
    return np.sum(psf, axis=(0, 1))


def energy_weights(xpsf: np.ndarray) -> np.ndarray:
    """Normalized 1/intensity weights (same as MATLAB)."""
    xpsf = np.asarray(xpsf, dtype=np.float64).ravel()
    energy = 1.0 / np.maximum(xpsf, np.finfo(np.float64).tiny)
    return energy / np.max(energy)


def apply_energy_modification(psf_list: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Parameters
    ----------
    psf_list : four arrays each (Y, X, Z)

    Returns
    -------
    psf_balanced : list of (Y, X, Z)
    xpsf : summed 4-view intensity curve (Z,)
    """
    curves = [xy_sum_curve(p) for p in psf_list]
    xpsf = sum(curves)
    w = energy_weights(xpsf)
    out = [p * w[np.newaxis, np.newaxis, :] for p in psf_list]
    return out, xpsf


def process_all_views_energy(
    psf_dir: Path | None = None,
    save: bool = True,
    run_seg: bool = True,
    seg_list: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Segment (optional) then energy-balance psf1–4; save psf{i}_energy.tif."""
    psf_dir = Path(psf_dir) if psf_dir is not None else SCRIPT_DIR
    if seg_list is None:
        if run_seg:
            seg_list = process_all_views(psf_dir=psf_dir, save=False)
        else:
            seg_list = []
            for v in range(1, 5):
                path = psf_dir / ("psf%d_seg.tif" % v)
                raw = tifffile.imread(str(path))
                if raw.shape[0] < raw.shape[1] and raw.shape[0] < raw.shape[2]:
                    raw = np.transpose(raw, (1, 2, 0))
                seg_list.append(np.asarray(raw, dtype=np.float64))

    balanced, xpsf = apply_energy_modification(seg_list)
    if save:
        for v, vol in enumerate(balanced, start=1):
            out_path = psf_dir / ("psf%d_energy.tif" % v)
            tifffile.imwrite(str(out_path), np.transpose(vol, (2, 0, 1)))
            print("saved %s shape=%s" % (out_path.name, vol.shape))
        np.save(str(psf_dir / "xpsf.npy"), xpsf)
    return balanced, xpsf


if __name__ == "__main__":
    process_all_views_energy()
