# -*- coding: utf-8 -*-
"""Merge four view-dictionaries (port of merge_dictionary1.m)."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from tplfm_utils import corrcoef_scalar


def _scaled_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    adis = np.asarray(c1, dtype=np.float64).ravel()[:3] - np.asarray(c2, dtype=np.float64).ravel()[:3]
    adis[0:2] *= 1.04
    adis[2] *= 0.27
    return float(np.sqrt(np.sum(adis**2)))


def _find_same(
    dict_a: list[dict[str, Any]],
    dict_b: list[dict[str, Any]],
    *,
    dist_thr: float = 12.0,
    corr_thr: float = 0.8,
    b_start: int = 0,
) -> np.ndarray:
    """Return (N,2) pairs [idx_in_a, idx_in_b] for matched neurons."""
    pairs = []
    for ia, na in enumerate(dict_a):
        for ib in range(b_start, len(dict_b)):
            nb = dict_b[ib]
            if _scaled_distance(na["center"], nb["center"]) >= dist_thr:
                continue
            if corrcoef_scalar(na["trace"], nb["trace"]) > corr_thr:
                pairs.append((ia, ib))
    if not pairs:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(pairs, dtype=int)


def _average_matched(
    base: list[dict[str, Any]],
    others: list[list[dict[str, Any]]],
    same_tables: list[np.ndarray],
) -> None:
    """
    For each neuron in `base` that matches any other dict, average center / center_allview.

    same_tables[k] rows are [idx_in_others[k], idx_in_base] (MATLAB Thesame* column order).
    """
    if not same_tables:
        return
    matched_base = set()
    for tab in same_tables:
        if tab.size:
            matched_base.update(int(x) for x in tab[:, 1])

    for ib in sorted(matched_base):
        positions = [np.asarray(base[ib]["center"], dtype=np.float64).ravel()[:3]]
        cavs = [np.asarray(base[ib]["center_allview"], dtype=np.float64)]
        for tab, od in zip(same_tables, others):
            if tab.size == 0:
                continue
            hits = np.flatnonzero(tab[:, 1] == ib)
            if hits.size == 0:
                continue
            ia = int(tab[int(hits[0]), 0])
            positions.append(np.asarray(od[ia]["center"], dtype=np.float64).ravel()[:3])
            cavs.append(np.asarray(od[ia]["center_allview"], dtype=np.float64))
        base[ib]["center"] = np.mean(np.stack(positions, axis=0), axis=0)
        # center_allview shapes (2,4) — stack along a temp axis then mean
        stacked = []
        for c in cavs:
            c = np.asarray(c, dtype=np.float64)
            if c.ndim == 3:
                c = c.reshape(2, -1)[:, :4] if c.shape[-1] >= 4 else c.squeeze()
            if c.shape == (2, 4):
                stacked.append(c)
            elif c.shape == (2, 1, 4):
                stacked.append(c[:, 0, :])
            else:
                stacked.append(c.reshape(2, 4))
        base[ib]["center_allview"] = np.mean(np.stack(stacked, axis=0), axis=0)


def _drop_indices(d: list[dict[str, Any]], drop: np.ndarray) -> list[dict[str, Any]]:
    drop_set = set(int(i) for i in np.asarray(drop).ravel()) if np.size(drop) else set()
    return [deepcopy(e) for i, e in enumerate(d) if i not in drop_set]


def merge_dictionary1(
    neuron_dictionary1: list[dict[str, Any]],
    neuron_dictionary2: list[dict[str, Any]],
    neuron_dictionary3: list[dict[str, Any]],
    neuron_dictionary4: list[dict[str, Any]],
    *,
    dist_thr: float = 12.0,
    corr_thr: float = 0.8,
) -> list[dict[str, Any]]:
    """Merge four per-view dictionaries; prefer dict1 then append unique from 2,3,4.

    dist_thr : max scaled 3D distance (MATLAB used 12 for ~0.75 um/px real data).
               For 0.285 um/px sim with residual parallax, use a larger value (e.g. 40–80).
    """
    d1 = [deepcopy(e) for e in neuron_dictionary1]
    d2 = [deepcopy(e) for e in neuron_dictionary2]
    d3 = [deepcopy(e) for e in neuron_dictionary3]
    d4 = [deepcopy(e) for e in neuron_dictionary4]

    same21 = _find_same(d2, d1, dist_thr=dist_thr, corr_thr=corr_thr)
    same31 = _find_same(d3, d1, dist_thr=dist_thr, corr_thr=corr_thr)
    same41 = _find_same(d4, d1, dist_thr=dist_thr, corr_thr=corr_thr)

    _average_matched(d1, [d2, d3, d4], [same21, same31, same41])

    d2 = _drop_indices(d2, same21[:, 0] if same21.size else np.array([], dtype=int))
    d3 = _drop_indices(d3, same31[:, 0] if same31.size else np.array([], dtype=int))
    d4 = _drop_indices(d4, same41[:, 0] if same41.size else np.array([], dtype=int))

    same32 = _find_same(d3, d2, dist_thr=dist_thr, corr_thr=corr_thr, b_start=0)
    same42 = _find_same(d4, d2, dist_thr=dist_thr, corr_thr=corr_thr, b_start=0)
    _average_matched(d2, [d3, d4], [same32, same42])

    merged = list(d1)
    merged.extend(d2)

    d3 = _drop_indices(d3, same32[:, 0] if same32.size else np.array([], dtype=int))
    d4 = _drop_indices(d4, same42[:, 0] if same42.size else np.array([], dtype=int))

    same43 = _find_same(d4, d3, dist_thr=dist_thr, corr_thr=corr_thr)
    _average_matched(d3, [d4], [same43])

    merged.extend(d3)
    d4 = _drop_indices(d4, same43[:, 0] if same43.size else np.array([], dtype=int))
    merged.extend(d4)
    return merged
