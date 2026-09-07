# -*- coding: utf-8 -*-
"""Estimate neuron depth from multi-view correlation (port of estimate_depth_other.m)."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from tplfm_utils import CONN4, corrcoef_scalar, parallax_dx_dy


def _align_views(views, move: np.ndarray):
    """Cubic-shift each view by move[v]=(dx,dy). Skip copy when move~0.

    Accepts ndarray (Y,X,T,V) or MultiViewMovies-like objects with .shape.
    """
    move = np.asarray(move, dtype=np.float64)
    if np.allclose(move, 0.0):
        return views
    # Materialize only when a real shift is required
    if not isinstance(views, np.ndarray):
        raise ValueError("non-zero move requires an in-memory ndarray views stack")
    out = np.empty(views.shape, dtype=np.float32)
    for v in range(views.shape[3]):
        out[:, :, :, v] = ndimage.shift(
            views[:, :, :, v].astype(np.float32, copy=False),
            shift=(-move[v, 1], -move[v, 0]),
            order=3,
            mode="constant",
            cval=0.0,
        )
    return out


def _mean_trace(views, yi: np.ndarray, xi: np.ndarray, v: int, n_t: int) -> np.ndarray:
    if yi.size == 0:
        return np.zeros(n_t, dtype=np.float64)
    if hasattr(views, "mean_trace"):
        return np.asarray(views.mean_trace(yi, xi, v), dtype=np.float64)
    return np.asarray(views[yi, xi, :, int(v)], dtype=np.float64).mean(axis=0)


def _sample_viewi_sum(viewi, yi: np.ndarray, xi: np.ndarray) -> np.ndarray:
    if yi.size == 0:
        return np.zeros(viewi.shape[2], dtype=np.float64)
    return np.asarray(viewi[yi, xi, :], dtype=np.float64).sum(axis=0)


def _sample_viewi_weights(viewi, yi: np.ndarray, xi: np.ndarray, tmax: int) -> np.ndarray:
    if yi.size == 0:
        return np.array([])
    return np.asarray(viewi[yi, xi, tmax], dtype=np.float64)


def _pix_to_index(
    xpix: np.ndarray,
    ypix: np.ndarray,
    h: int,
    w: int,
    *,
    pix_one_based: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_coord, y_coord, xi, yi) for indexing viewi[yi, xi]."""
    if pix_one_based:
        x = np.clip(xpix, 1, w)
        y = np.clip(ypix, 1, h)
        xi = np.rint(x).astype(int) - 1
        yi = np.rint(y).astype(int) - 1
    else:
        x = np.clip(xpix, 0, w - 1)
        y = np.clip(ypix, 0, h - 1)
        xi = np.rint(x).astype(int)
        yi = np.rint(y).astype(int)
    return x, y, xi, yi


def estimate_depth_other(
    stat: list[dict[str, Any]],
    F: np.ndarray,
    Fneu: np.ndarray,
    Order_neuron: np.ndarray,
    coef_psf_xz: np.ndarray,
    coef_psf_yz: np.ndarray,
    viewi: np.ndarray,
    views: np.ndarray,
    view: int,
    move: np.ndarray,
    factor: float = 5.0,
    depth_min: int = -50,
    depth_max: int = 50,
    depth_step: int = 1,
    *,
    pix_one_based: bool = False,
    skip_remask: bool = True,
    x0: float = 60.0,
    y0: float = 60.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list,
    list[dict[str, np.ndarray]],
    np.ndarray,
]:
    """
    Parameters
    ----------
    stat : suite2p ROIs with xpix/ypix
    pix_one_based : True for MATLAB suite2p (1-based); False for Python suite2p (0-based)
    Order_neuron : 1-based ROI indices into stat
    view : primary view index 1..4 (MATLAB)
    factor : zoom scale (5 for real zoom15-PSF/zoom3-data; use 1 for matched sim)
    """
    view0 = int(view) - 1
    Order_neuron = np.asarray(Order_neuron).astype(int).ravel()
    order0 = Order_neuron - 1

    views = _align_views(views, move)
    # Do not cast full movies to float64 — keep memmap / float32

    h, w, n_t, n_views = views.shape
    if hasattr(viewi, "shape"):
        if viewi.shape[0] != h or viewi.shape[1] != w:
            raise ValueError("viewi spatial size must match views")
        if viewi.shape[2] != n_t:
            n_t = min(n_t, int(viewi.shape[2]))
    n_neu = order0.size
    info = np.array([0, 1, 2, 3], dtype=int)
    info1 = np.delete(info, view0)

    _ = F
    _ = Fneu

    rawtrace = np.zeros((n_neu, n_t), dtype=np.float64)
    neuron_centeri = np.zeros((n_neu, 3), dtype=np.float64)
    Masks: list = [None] * n_neu

    sig = 12.0
    yy, xx = np.mgrid[0:49, 0:51]
    filt = np.minimum(np.e * np.exp(-((xx - 24.0) ** 2 + (yy - 25.0) ** 2) / sig**2), 1.0)

    lo_x, hi_x = (1, w) if pix_one_based else (0, w - 1)
    lo_y, hi_y = (1, h) if pix_one_based else (0, h - 1)

    for num in range(n_neu):
        st = stat[order0[num]]
        xpix = np.asarray(st["xpix"], dtype=np.float64).ravel() - move[view0, 1]
        ypix = np.asarray(st["ypix"], dtype=np.float64).ravel() - move[view0, 0]
        xpix, ypix, xi, yi = _pix_to_index(xpix, ypix, h, w, pix_one_based=pix_one_based)

        if xi.size:
            rawtrace[num] = _sample_viewi_sum(viewi, yi, xi)

        tmax = int(np.argmax(rawtrace[num]))
        weights = _sample_viewi_weights(viewi, yi, xi, tmax)
        wsum = float(weights.sum()) if weights.size else 0.0
        if wsum <= 0:
            neuron_centeri[num, 0] = float(xpix.mean())
            neuron_centeri[num, 1] = float(ypix.mean())
        else:
            neuron_centeri[num, 0] = float(np.sum(xpix * weights) / wsum)
            neuron_centeri[num, 1] = float(np.sum(ypix * weights) / wsum)

        if skip_remask:
            Masks[num] = (np.array([]), np.array([]))
            continue

        cy = neuron_centeri[num, 1]
        cx = neuron_centeri[num, 0]
        # remask path only for in-memory arrays
        viewi_arr = np.asarray(viewi)
        viewi00 = np.pad(viewi_arr, ((40, 40), (40, 40), (0, 0)), mode="constant")
        r0 = int(round(cy)) + 40 - 24
        c0 = int(round(cx)) + 40 - 25
        patch = viewi00[r0 : r0 + 49, c0 : c0 + 51, :].astype(np.float64)
        if patch.shape[0] != 49 or patch.shape[1] != 51:
            Masks[num] = (np.array([]), np.array([]))
            continue
        max_time = patch[:, :, min(tmax, patch.shape[2] - 1)]
        cori_map = np.zeros((49, 51), dtype=np.float64)
        for xr in range(49):
            for yr in range(51):
                cori_map[xr, yr] = corrcoef_scalar(rawtrace[num], patch[xr, yr, :])
        aa = cori_map * max_time
        aa = np.nan_to_num(aa, nan=-1.0) * filt
        thr = float(np.max(aa[14:35, 15:36])) / 2.0 if aa.size else 0.0
        aa = aa.copy()
        aa[aa < thr] = 0.0
        labeled, n_lab = ndimage.label(aa > 0, structure=CONN4)
        if n_lab > 0:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            ind = int(np.argmax(counts))
            aa[labeled != ind] = 0.0
            ys, xs = np.nonzero(aa > 0)
        else:
            ys, xs = np.array([], dtype=int), np.array([], dtype=int)

        ax = ys.astype(np.float64) + 1 + cy - 25
        ay = xs.astype(np.float64) + 1 + cx - 26
        Masks[num] = (ay, ax)

    n_depth = len(range(depth_min, depth_max + 1, depth_step))
    cori_allneuron_allz = np.zeros((n_depth, n_neu), dtype=np.float64)
    pixels: list[dict[str, np.ndarray]] = []

    for num in range(n_neu):
        st = stat[order0[num]]
        xpix = np.asarray(st["xpix"], dtype=np.float64).ravel() - move[view0, 1]
        ypix = np.asarray(st["ypix"], dtype=np.float64).ravel() - move[view0, 0]
        d1 = xpix - neuron_centeri[num, 0]
        d2 = ypix - neuron_centeri[num, 1]
        pixels.append({"neuron_pixels_delta1": d1.copy(), "neuron_pixels_delta2": d2.copy()})

        cori_allz = np.zeros(n_depth, dtype=np.float64)
        for iz, nz in enumerate(range(depth_min, depth_max + 1, depth_step)):
            dx, dy = parallax_dx_dy(
                nz, info[view0], coef_psf_xz, coef_psf_yz, factor, x0=x0, y0=y0
            )
            refer = np.array(
                [neuron_centeri[num, 0] - dx, neuron_centeri[num, 1] - dy], dtype=np.float64
            )
            cori_this = 0.0
            for v_other in info1:
                dx, dy = parallax_dx_dy(
                    nz, int(v_other), coef_psf_xz, coef_psf_yz, factor, x0=x0, y0=y0
                )
                guess = refer + np.array([dx, dy])
                px = np.clip(d1 + guess[0], lo_x, hi_x)
                py = np.clip(d2 + guess[1], lo_y, hi_y)
                if pix_one_based:
                    xi = np.rint(px).astype(int) - 1
                    yi = np.rint(py).astype(int) - 1
                else:
                    xi = np.rint(px).astype(int)
                    yi = np.rint(py).astype(int)
                trace_v = _mean_trace(views, yi, xi, int(v_other), n_t)
                r = corrcoef_scalar(rawtrace[num], trace_v)
                if np.isfinite(r):
                    cori_this += r
            cori_allz[iz] = cori_this
        cori_allneuron_allz[:, num] = cori_allz

    centers_allview = np.zeros((2, n_neu, 4), dtype=np.float64)
    depth_neuron = np.zeros(n_neu, dtype=np.float64)

    for ii in range(n_neu):
        peak = cori_allneuron_allz[:, ii]
        finite = np.isfinite(peak)
        if not np.any(finite):
            depth_neuron[ii] = 0.0
        else:
            peak_safe = np.where(finite, peak, -np.inf)
            depth_neuron[ii] = float(
                np.mean(np.flatnonzero(peak_safe == peak_safe.max())) * depth_step
                + depth_min
            )
        dx, dy = parallax_dx_dy(
            depth_neuron[ii], info[view0], coef_psf_xz, coef_psf_yz, factor, x0=x0, y0=y0
        )
        neuron_centeri[ii, 0] -= dx
        neuron_centeri[ii, 1] -= dy
        neuron_centeri[ii, 2] = depth_neuron[ii]
        centers_allview[0, ii, view0] = neuron_centeri[ii, 0]
        centers_allview[1, ii, view0] = neuron_centeri[ii, 1]
        for vn in range(1, n_views):
            v_other = info1[vn - 1]
            dx, dy = parallax_dx_dy(
                depth_neuron[ii], v_other, coef_psf_xz, coef_psf_yz, factor, x0=x0, y0=y0
            )
            centers_allview[0, ii, v_other] = neuron_centeri[ii, 0] + dx
            centers_allview[1, ii, v_other] = neuron_centeri[ii, 1] + dy

    return (
        neuron_centeri,
        cori_allneuron_allz,
        depth_neuron,
        rawtrace,
        Masks,
        pixels,
        centers_allview,
    )
