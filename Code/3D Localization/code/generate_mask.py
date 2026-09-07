# -*- coding: utf-8 -*-
"""Build multi-view reconstruction mask (port of generate_mask.m)."""
from __future__ import annotations

from typing import Any

import numpy as np

from tplfm_utils import parallax_dx_dy


def generate_mask(
    neuron_dictionary: list[dict[str, Any]],
    Info: list[int] | np.ndarray | None = None,
    coef_psf_xz: np.ndarray | None = None,
    coef_psf_yz: np.ndarray | None = None,
    xpsf: np.ndarray | None = None,
    fov: int = 255,
    factor: float = 5.0,
    energy_len: int = 80,
    x0: float = 60.0,
    y0: float = 60.0,
) -> np.ndarray:
    """
    Parameters
    ----------
    neuron_dictionary : merged neuron list
    Info : 1-based view indices, default [1,2,3,4]
    coef_psf_xz, coef_psf_yz : (2, n_view) from psf_fit
    xpsf : per-plane intensity sum curve (from energy modification)
    fov : lateral size (MATLAB hard-coded 255)
    """
    if Info is None:
        Info = [1, 2, 3, 4]
    Info = [int(v) for v in np.asarray(Info).ravel()]
    n_views = len(Info)
    if coef_psf_xz is None or coef_psf_yz is None or xpsf is None:
        raise ValueError("coef_psf_xz, coef_psf_yz, and xpsf are required")

    xpsf = np.asarray(xpsf, dtype=np.float64).ravel()
    zi = np.linspace(0, xpsf.size - 1, energy_len)
    xpsf_i = np.interp(zi, np.arange(xpsf.size, dtype=np.float64), xpsf)
    psfenergy1 = (1.0 / np.maximum(xpsf_i, np.finfo(np.float64).tiny))
    psfenergy1 = psfenergy1 / np.max(psfenergy1)

    nmask = np.zeros((fov, fov, n_views), dtype=np.float64)

    for entry in neuron_dictionary:
        center_allview = np.asarray(entry["center_allview"], dtype=np.float64)
        if center_allview.ndim == 3:
            center_allview = center_allview[:, 0, :]
        center = np.asarray(entry["center"], dtype=np.float64).ravel()
        pixels_delta1 = np.asarray(entry["pixels1"], dtype=np.float64).ravel()
        pixels_delta2 = np.asarray(entry["pixels2"], dtype=np.float64).ravel()
        nz = float(center[2])
        nz1 = int(round((nz + 40.0) / 2.0) + 1) - 1
        nz1 = int(np.clip(nz1, 0, energy_len - 1))
        w = float(psfenergy1[nz1])

        for vn, view_1b in enumerate(Info):
            v0 = view_1b - 1
            dx, dy = parallax_dx_dy(nz, v0, coef_psf_xz, coef_psf_yz, factor, x0=x0, y0=y0)
            cx = float(center_allview[0, vn]) + dx
            cy = float(center_allview[1, vn]) + dy
            px = pixels_delta1 + cx
            py = pixels_delta2 + cy
            px = np.clip(px, 1, fov)
            py = np.clip(py, 1, fov)
            xi = np.rint(px).astype(int) - 1
            yi = np.rint(py).astype(int) - 1
            for ii in range(xi.size):
                nmask[yi[ii], xi[ii], vn] = w

    return nmask
