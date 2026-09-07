# -*- coding: utf-8 -*-
"""Build per-neuron dictionary (port of build_dictionary.m)."""
from __future__ import annotations

from typing import Any

import numpy as np


def build_dictionary(
    neuron_centeri: np.ndarray,
    rawtrace: np.ndarray,
    centers_allview: np.ndarray,
    pixelss: list[dict[str, np.ndarray]],
    cori_allneuron_allz_view: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Parameters
    ----------
    neuron_centeri : (N, 3) x,y,z
    rawtrace : (N, T)
    centers_allview : (2, N, 4)
    pixelss : list of dicts with neuron_pixels_delta1 / neuron_pixels_delta2
    cori_allneuron_allz_view : (n_z, N) or (n_z,)

    Returns
    -------
    neuron_dictionary : list of neuron dicts
    """
    neuron_centeri = np.asarray(neuron_centeri, dtype=np.float64)
    rawtrace = np.asarray(rawtrace, dtype=np.float64)
    centers_allview = np.asarray(centers_allview, dtype=np.float64)
    cori = np.asarray(cori_allneuron_allz_view, dtype=np.float64)
    n = neuron_centeri.shape[0]
    out: list[dict[str, Any]] = []
    for ii in range(n):
        px = pixelss[ii]
        entry = {
            "center": neuron_centeri[ii].copy(),
            "trace": rawtrace[ii].copy(),
            "center_allview": centers_allview[:, ii, :].copy(),  # (2, 4)
            "pixels1": np.asarray(px["neuron_pixels_delta1"], dtype=np.float64).copy(),
            "pixels2": np.asarray(px["neuron_pixels_delta2"], dtype=np.float64).copy(),
            "cori_allneuron_allz": cori[:, ii].copy() if cori.ndim == 2 else cori.copy(),
        }
        out.append(entry)
    return out
