# -*- coding: utf-8 -*-
"""Official step: give unmatched recon ROIs view-suite2p footprints, then rematch.

After method1 localize + Hungarian match, unmatched m2 cells have no view ROI.
This appends (or updates) one recon-warped, parallax-placed iscell ROI in each
of the four view suite2p folders, localizes those footprints, and writes a
trusted match when corr / d_xy / d_z pass the same gates as recover.

Does not replace original view detections. Already-trusted matches are skipped.
"""
from __future__ import annotations

import csv
import gc
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from build_dictionary import build_dictionary
from dataset_config import (
    OUT_DIR,
    PIXEL_UM,
    PLANE_Z_BASE,
    T_STRIDE,
    move_divisor_from_coef,
    view_plane_dir,
)
from estimate_depth_other import estimate_depth_other
from match_linear_runner_to_recon_planes import (
    M1_DATA,
    W_R,
    W_XY,
    W_Z,
    apply_xy,
    load_matches_csv,
    load_method2,
)
from recover_m2_from_bin import (
    TRUST_D_XY_PX,
    TRUST_D_Z_UM,
    TRUST_R_MIN,
    CropViews,
    eval_views_at,
    footprint_at_center,
    inverse_xy,
    load_xy_tfm,
    open_memmaps,
    recover_one,
    resolve_match_dir,
    search_boxes,
    view_center_in_bin,
    warped_m2_footprint,
)
from tplfm_utils import compute_move as _compute_move, corrcoef_scalar

DEPTH_STEP_UM = 2
FILTER_MIN_PEAK_CORR = 0.30
MARGIN = 90
CHUNK = 80
S2P_KEYS = ("stat.npy", "F.npy", "Fneu.npy", "iscell.npy", "spks.npy", "ops.npy")

BACKUP = OUT_DIR / "views_suite2p_original_backup"
MAP_PATH = OUT_DIR / "views_suite2p_from_recon" / "appended_unmatched.json"
DICT_M1 = OUT_DIR / "neuron_dictionary.npy"
LOC_CSV = OUT_DIR / "localization.csv"
CATALOG = OUT_DIR / "match" / "m2_3d_coverage" / "m2_3d_catalog.csv"
DICT_M2 = OUT_DIR / "match" / "m2_3d_coverage" / "m2_3d_dictionary.npy"
COV_JSON = OUT_DIR / "match" / "m2_3d_coverage" / "coverage_summary.json"


class AlignedCropViews:
    def __init__(self, crops, origins, move_i, ly, lx, n_t):
        self.crops = crops
        self.origins = origins
        self.move_i = np.asarray(move_i, dtype=int)
        self.shape = (int(ly), int(lx), int(n_t), 4)

    def _raw_xy(self, yi, xi, v):
        ys = np.asarray(yi, dtype=np.int64).ravel() + int(self.move_i[v, 0])
        xs = np.asarray(xi, dtype=np.int64).ravel() + int(self.move_i[v, 1])
        return ys, xs

    def _gather(self, v, ys, xs, t=None):
        crop = self.crops[v]
        y0, x0 = self.origins[v]
        n_t, bh, bw = crop.shape
        ry = ys - y0
        rx = xs - x0
        ok = (ry >= 0) & (ry < bh) & (rx >= 0) & (rx < bw)
        if t is None:
            out = np.zeros((ys.size, n_t), dtype=np.float64)
            if ok.any():
                out[ok] = crop[:, ry[ok], rx[ok]].T
            return out
        out = np.zeros(ys.size, dtype=np.float64)
        tt = int(np.clip(int(t), 0, n_t - 1))
        if ok.any():
            out[ok] = crop[tt, ry[ok], rx[ok]]
        return out

    def mean_trace(self, yi, xi, v):
        ys, xs = self._raw_xy(yi, xi, v)
        pix = self._gather(v, ys, xs, t=None)
        if pix.size == 0:
            return np.zeros(self.shape[2], dtype=np.float64)
        return pix.mean(axis=0)


class ViewISlice:
    def __init__(self, parent: AlignedCropViews, v: int):
        self.p = parent
        self.v = int(v)
        ly, lx, n_t, _ = parent.shape
        self.shape = (ly, lx, n_t)

    def __getitem__(self, key):
        yi, xi, t = key
        ys, xs = self.p._raw_xy(yi, xi, self.v)
        if isinstance(t, slice):
            return self.p._gather(self.v, ys, xs, t=None)
        return self.p._gather(self.v, ys, xs, t=t)


def _center_spread(cav: np.ndarray) -> float:
    pts = [
        (float(cav[0, v]), float(cav[1, v]))
        for v in range(4)
        if np.isfinite(cav[0, v]) and np.isfinite(cav[1, v])
    ]
    dmax = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
            if d > dmax:
                dmax = d
    return dmax


def _make_stat(template: dict, xs, ys) -> dict:
    skip = {"xpix", "ypix", "lam", "med", "npix", "neuropil_mask", "soma_crop"}
    st = {k: template[k] for k in template if k not in skip}
    lam = np.ones(xs.size, dtype=np.float32)
    st["xpix"] = xs
    st["ypix"] = ys
    st["lam"] = lam
    st["med"] = [int(np.rint(float(ys.mean()))), int(np.rint(float(xs.mean())))]
    st["npix"] = int(xs.size)
    dx = xs.astype(np.float64) - float(xs.mean())
    dy = ys.astype(np.float64) - float(ys.mean())
    st["radius"] = float(np.sqrt(np.max(dx * dx + dy * dy) + 1.0))
    return st


def _extract_f(mem, xs, ys, nf: int) -> np.ndarray:
    tr = np.zeros(nf, dtype=np.float32)
    for s in range(0, nf, CHUNK):
        e = min(s + CHUNK, nf)
        block = np.asarray(mem[s:e], dtype=np.float32)
        tr[s:e] = block[:, ys, xs].mean(axis=1)
        del block
    return tr


def _backup_view_suite2p() -> None:
    if (BACKUP / "view_1" / "stat.npy").is_file():
        return
    BACKUP.mkdir(parents=True, exist_ok=True)
    (BACKUP / "README.txt").write_text(
        "Original Views/{1-4}/suite2p/plane0 npy files (not data.bin).\n",
        encoding="utf-8",
    )
    for v in range(1, 5):
        src = view_plane_dir(v, M1_DATA)
        dst = BACKUP / ("view_%d" % v)
        dst.mkdir(parents=True, exist_ok=True)
        for name in S2P_KEYS:
            p = src / name
            if p.is_file():
                shutil.copy2(str(p), str(dst / name))
    print("backed up original view suite2p ->", BACKUP, flush=True)


def _save_map(mp: dict[str, Any]) -> None:
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(mp, indent=2), encoding="utf-8")


def _write_unmatched(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _load_map() -> dict[str, Any]:
    if MAP_PATH.is_file():
        return json.loads(MAP_PATH.read_text(encoding="utf-8"))
    legacy = OUT_DIR / "views_suite2p_from_recon" / "appended_recon_roi.json"
    out: dict[str, Any] = {"appended": []}
    if legacy.is_file():
        old = json.loads(legacy.read_text(encoding="utf-8"))
        out["appended"].append(
            {
                "m2_plane": int(old["m2_plane"]),
                "m2_roi": int(old["m2_roi"]),
                "view_roi_idx": {str(k): int(v) for k, v in old["new_view_roi_idx"].items()},
                "promoted": True,
            }
        )
    return out


def _map_entry(mp: dict[str, Any], plane: int, roi: int) -> dict[str, Any] | None:
    for e in mp.get("appended", []):
        if int(e["m2_plane"]) == plane and int(e["m2_roi"]) == roi:
            return e
    return None


def _write_view_roi(
    v: int,
    roi_i: int | None,
    xs: np.ndarray,
    ys: np.ndarray,
    mem,
    nf: int,
) -> int:
    plane = view_plane_dir(v, M1_DATA)
    stat = list(np.load(str(plane / "stat.npy"), allow_pickle=True))
    F0 = np.load(str(plane / "F.npy"))
    Fneu = np.load(str(plane / "Fneu.npy"))
    spks = np.load(str(plane / "spks.npy"))
    iscell = np.load(str(plane / "iscell.npy"))
    ops = np.load(str(plane / "ops.npy"), allow_pickle=True).item()
    frow = _extract_f(mem, xs, ys, nf)
    st = _make_stat(dict(stat[0]), xs, ys)
    if roi_i is None or roi_i >= len(stat):
        stat.append(st)
        F0 = np.concatenate([F0, frow[None, :]], axis=0)
        zrow = np.zeros((1, F0.shape[1]), dtype=Fneu.dtype)
        Fneu = np.concatenate([Fneu, zrow], axis=0)
        spks = np.concatenate([spks, np.zeros_like(zrow)], axis=0)
        if iscell.ndim == 2:
            iscell = np.vstack([iscell, np.array([[1.0, 1.0]], dtype=iscell.dtype)])
        else:
            iscell = np.concatenate([iscell, np.array([1], dtype=iscell.dtype)])
        roi_i = len(stat) - 1
    else:
        stat[roi_i] = st
        F0[roi_i] = frow
        if Fneu.ndim == 2 and roi_i < Fneu.shape[0]:
            Fneu[roi_i] = 0
    ops = dict(ops)
    ops["nrois"] = int(len(stat))
    np.save(str(plane / "stat.npy"), np.array(stat, dtype=object))
    np.save(str(plane / "F.npy"), F0)
    np.save(str(plane / "Fneu.npy"), Fneu)
    np.save(str(plane / "spks.npy"), spks)
    np.save(str(plane / "iscell.npy"), iscell)
    np.save(str(plane / "ops.npy"), ops)
    return int(roi_i)


def _localize_appended(
    view_idx: dict[int, int],
    stats,
    Fs,
    Fneus,
    views: AlignedCropViews,
    move_i,
    coef_xz,
    coef_yz,
    factor,
    x00,
    y00,
    depth_min,
    depth_max,
    stride_rs: np.ndarray,
) -> tuple[dict[str, Any], float, float, int]:
    candidates = []
    for view in range(1, 5):
        st_list = []
        for st in stats[view]:
            st2 = dict(st)
            st2["xpix"] = np.asarray(st["xpix"], dtype=np.float64) - float(move_i[view - 1, 1])
            st2["ypix"] = np.asarray(st["ypix"], dtype=np.float64) - float(move_i[view - 1, 0])
            st_list.append(st2)
        roi_i = view_idx[view]
        (
            neuron_centeri,
            cori,
            _depth,
            rawtrace,
            _masks,
            pixels,
            centers_allview,
        ) = estimate_depth_other(
            st_list,
            Fs[view],
            Fneus[view],
            np.array([roi_i + 1], dtype=int),
            coef_xz,
            coef_yz,
            ViewISlice(views, view - 1),
            views,
            view,
            np.zeros((4, 2), dtype=np.float64),
            factor=factor,
            pix_one_based=False,
            x0=x00,
            y0=y00,
            depth_min=depth_min,
            depth_max=depth_max,
            depth_step=DEPTH_STEP_UM,
        )
        d = build_dictionary(neuron_centeri, rawtrace, centers_allview, pixels, cori)[0]
        peak = float(np.nanmax(cori[:, 0])) if cori.size else float("nan")
        d["peak_corr"] = peak
        d["primary_view"] = view
        candidates.append((d, float(stride_rs[view - 1]), peak))
    usable = [
        c
        for c in candidates
        if np.isfinite(c[2]) and c[2] >= FILTER_MIN_PEAK_CORR and np.isfinite(c[1])
    ]
    if not usable:
        usable = candidates
    d, corr, peak = max(usable, key=lambda c: c[1])
    return d, corr, peak, int(d["primary_view"])


def _promote_files(
    m2: dict[str, Any],
    d: dict[str, Any],
    col: float,
    row: float,
    z: float,
    corr: float,
    peak: float,
    p_view: int,
    d_xy: float,
    d_z: float,
    x1m: float,
    y1m: float,
    trace: np.ndarray,
    match_csv: Path,
) -> int:
    m1_dict = list(np.load(str(DICT_M1), allow_pickle=True))
    m1_idx = len(m1_dict)
    cav = np.asarray(d["center_allview"], dtype=np.float64)
    if cav.shape != (2, 4):
        cav = cav.reshape(2, 4)
    entry = {
        "center": np.array([col, row, z], dtype=np.float64),
        "center_allview": cav.copy(),
        "pixels1": np.asarray(d["pixels1"], dtype=np.float64).copy(),
        "pixels2": np.asarray(d["pixels2"], dtype=np.float64).copy(),
        "trace": np.asarray(trace, dtype=np.float64).copy(),
        "cori_allneuron_allz": np.asarray(d.get("cori_allneuron_allz", []), dtype=np.float64).copy(),
        "primary_view": p_view,
        "peak_corr": float(peak),
        "source": "matched_trusted",
        "m1_idx": m1_idx,
        "m2_plane": int(m2["plane_id"]),
        "m2_roi": int(m2["roi_idx"]),
    }
    m1_dict.append(entry)
    np.save(str(DICT_M1), np.array(m1_dict, dtype=object))

    loc_rows = list(csv.DictReader(open(LOC_CSV, encoding="utf-8")))
    loc_fields = list(loc_rows[0].keys())
    loc_rows.append(
        {
            "idx": str(m1_idx),
            "orig_idx": str(m1_idx),
            "col": col,
            "row": row,
            "z_um": z,
            "z_plane_equiv": z / 2.0,
            "x_um": col * PIXEL_UM,
            "y_um": row * PIXEL_UM,
            "n_views_matched": 4,
            "peak_corr": peak,
            "center_spread_px": _center_spread(cav),
            "mean_trace_corr": corr,
            "min_trace_corr": corr,
            "primary_view": p_view,
        }
    )
    with open(LOC_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=loc_fields)
        w.writeheader()
        w.writerows(loc_rows)

    match_rows = list(csv.DictReader(open(match_csv, encoding="utf-8")))
    match_fields = list(match_rows[0].keys())
    match_rows.append(
        {
            "m1_idx": m1_idx,
            "m2_plane": int(m2["plane_id"]),
            "m2_roi": int(m2["roi_idx"]),
            "col1": col,
            "row1": row,
            "z1": z,
            "x2": m2["x"],
            "y2": m2["y"],
            "z2": m2["z_um"],
            "x1_in_m2": x1m,
            "y1_in_m2": y1m,
            "d_xy_px": d_xy,
            "d_z_um": d_z,
            "corr": corr,
            "cost": W_XY * d_xy + W_Z * d_z + W_R * (1.0 - corr),
        }
    )
    with open(match_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=match_fields)
        w.writeheader()
        w.writerows(match_rows)

    if CATALOG.is_file():
        cat = list(csv.DictReader(open(CATALOG, encoding="utf-8")))
        for r in cat:
            if int(r["m2_plane"]) == int(m2["plane_id"]) and int(r["m2_roi"]) == int(m2["roi_idx"]):
                r["source"] = "matched_trusted"
                r["col"] = col
                r["row"] = row
                r["z_um"] = z
                r["corr_or_best_r"] = corr
                r["primary_view"] = p_view
                r["m1_idx"] = m1_idx
                r["d_xy_px"] = d_xy
                r["d_z_um"] = d_z
                r["n_pass"] = ""
                r["accepted"] = 1
        with open(CATALOG, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cat[0].keys()))
            w.writeheader()
            w.writerows(cat)

    if DICT_M2.is_file():
        m2_dict = list(np.load(str(DICT_M2), allow_pickle=True))
        found = False
        for e in m2_dict:
            if int(e.get("m2_plane", -1)) == int(m2["plane_id"]) and int(
                e.get("m2_roi", -1)
            ) == int(m2["roi_idx"]):
                e.update(entry)
                e["source"] = "matched_trusted"
                e["m1_idx"] = m1_idx
                e["m2_plane"] = int(m2["plane_id"])
                e["m2_roi"] = int(m2["roi_idx"])
                e["center"] = np.array([col, row, z], dtype=np.float64)
                found = True
                break
        if not found:
            m2_dict.append(dict(entry))
        np.save(str(DICT_M2), np.array(m2_dict, dtype=object))

    if COV_JSON.is_file():
        cov = json.loads(COV_JSON.read_text(encoding="utf-8"))
        src = cov.setdefault("by_source", {})
        src["matched_trusted"] = int(src.get("matched_trusted", 0)) + 1
        for rec_key in ("recovered_bin_refined", "recovered_bin"):
            if rec_key in src and int(src[rec_key]) > 0:
                src[rec_key] = int(src[rec_key]) - 1
                break
        COV_JSON.write_text(json.dumps(cov, indent=2), encoding="utf-8")

    return m1_idx


def main() -> None:
    match_dir = resolve_match_dir()
    match_csv = match_dir / "matches.csv"
    unmatch_csv = match_dir / "unmatched_m2.csv"
    if not match_csv.is_file():
        raise SystemExit("no matches.csv under %s" % match_dir)
    if not unmatch_csv.is_file():
        print("no unmatched_m2.csv; nothing to promote")
        return

    matches = load_matches_csv(match_csv)
    trusted_keys = {
        (int(m["m2_plane"]), int(m["m2_roi"]))
        for m in matches
        if float(m["corr"]) >= TRUST_R_MIN
        and float(m["d_xy_px"]) <= TRUST_D_XY_PX
        and float(m["d_z_um"]) <= TRUST_D_Z_UM
    }
    um_rows = list(csv.DictReader(open(unmatch_csv, encoding="utf-8")))
    um_fields = list(um_rows[0].keys()) if um_rows else ["m2_plane", "m2_roi", "x2", "y2", "z2"]
    candidates = [
        (int(r["m2_plane"]), int(r["m2_roi"]))
        for r in um_rows
        if (int(r["m2_plane"]), int(r["m2_roi"])) not in trusted_keys
    ]
    print(
        "unmatched=%d  already trusted=%d  to try=%d"
        % (len(um_rows), len(trusted_keys), len(candidates)),
        flush=True,
    )
    if not candidates:
        print("no unmatched m2 left to promote")
        return

    m2_pool = load_method2()
    m2_map = {(m["plane_id"], m["roi_idx"]): m for m in m2_pool}
    tfm = load_xy_tfm(match_dir)
    coef = np.load(str(OUT_DIR / "psf_fit_coef.npz"))
    coef_xz = np.asarray(coef["coef_psf_xz"], dtype=np.float64)
    coef_yz = np.asarray(coef["coef_psf_yz"], dtype=np.float64)
    factor = float(coef["factor"]) if "factor" in coef.files else 2.5
    x00, y00 = float(coef["x0"]), float(coef["y0"])
    move_i = np.round(
        _compute_move(
            np.asarray(coef["centers_1based"], dtype=np.float64),
            factor=move_divisor_from_coef(coef),
        )
    ).astype(int)
    z_um = np.asarray(coef["z_um"], dtype=np.float64)
    half_span = float(max(abs(z_um[0]), abs(z_um[-1])))
    n_steps = int(np.ceil(half_span / DEPTH_STEP_UM))
    depth_min = -n_steps * DEPTH_STEP_UM
    depth_max = n_steps * DEPTH_STEP_UM

    _backup_view_suite2p()
    mp = _load_map()
    mems, ly, lx, nf = open_memmaps()
    stride = max(int(T_STRIDE), 1)
    promoted_keys: set[tuple[int, int]] = set()
    n_skip_r = 0

    for k, key in enumerate(candidates):
        if k > 0 and k % 5 == 0:
            del mems
            gc.collect()
            mems, ly, lx, nf = open_memmaps()
        if key not in m2_map:
            print("  skip p%d/roi%d (not in pool)" % key, flush=True)
            continue
        m2 = m2_map[key]
        print("  [%d/%d] p%d/roi%d ..." % (k + 1, len(candidates), key[0], key[1]), flush=True)
        seed = warped_m2_footprint(m2, tfm, ly, lx)
        if seed is None:
            print("    no warped footprint", flush=True)
            continue
        col0, row0 = inverse_xy(float(m2["x"]), float(m2["y"]), tfm)
        res = recover_one(
            m2, tfm, mems, ly, lx, nf, move_i, coef_xz, coef_yz, factor, x00, y00
        )
        z_place = float(res["best_z"])
        p0_place = int(res["primary_v0"])
        fi = np.arange(0, min(nf, int(np.asarray(m2["trace"]).size)), stride, dtype=int)
        tr_m2 = np.asarray(m2["trace"], dtype=np.float64)
        boxes = search_boxes(
            col0, row0, z_place, move_i, coef_xz, coef_yz, factor, x00, y00, ly, lx, seed
        )
        crops_eval = CropViews(mems, boxes, fi)
        rs_place, _tr, rmax_place, _score = eval_views_at(
            col0,
            row0,
            z_place,
            p0_place,
            crops_eval,
            ly,
            lx,
            fi,
            tr_m2[fi],
            move_i,
            coef_xz,
            coef_yz,
            factor,
            x00,
            y00,
            seed,
        )
        print(
            "    activity z=%.1f p0=v%d  seed-pose rmax=%.3f rs=%s"
            % (z_place, p0_place + 1, rmax_place, np.round(rs_place, 3)),
            flush=True,
        )
        if not np.isfinite(rmax_place) or rmax_place < TRUST_R_MIN:
            n_skip_r += 1
            continue

        placed = []
        for v0 in range(4):
            cx, cy = view_center_in_bin(
                col0, row0, z_place, p0_place, v0, move_i, coef_xz, coef_yz, factor, x00, y00
            )
            xs, ys = footprint_at_center(seed[0], seed[1], cx, cy, ly, lx)
            placed.append((xs, ys, cx, cy))

        prev = _map_entry(mp, key[0], key[1])
        view_idx: dict[int, int] = {}
        for v0 in range(4):
            old_i = None
            if prev is not None:
                old_i = int(prev["view_roi_idx"][str(v0 + 1)])
            xs, ys, _cx, _cy = placed[v0]
            view_idx[v0 + 1] = _write_view_roi(v0 + 1, old_i, xs, ys, mems[v0], nf)

        stats, Fs, Fneus = {}, {}, {}
        stride_rs = np.full(4, np.nan)
        for v in range(1, 5):
            plane = view_plane_dir(v, M1_DATA)
            stats[v] = list(np.load(str(plane / "stat.npy"), allow_pickle=True))
            Fs[v] = np.load(str(plane / "F.npy"))
            Fneus[v] = np.load(str(plane / "Fneu.npy"))
            f = np.asarray(Fs[v][view_idx[v]], dtype=np.float64)
            stride_rs[v - 1] = corrcoef_scalar(f[fi], tr_m2[fi])

        crops, origins = [], []
        for v0 in range(4):
            xs, ys, cx, cy = placed[v0]
            x0b = int(max(0, np.floor(cx - MARGIN)))
            x1b = int(min(lx, np.ceil(cx + MARGIN) + 1))
            y0b = int(max(0, np.floor(cy - MARGIN)))
            y1b = int(min(ly, np.ceil(cy + MARGIN) + 1))
            crops.append(np.asarray(mems[v0][fi, y0b:y1b, x0b:x1b], dtype=np.float32))
            origins.append((y0b, x0b))
        views = AlignedCropViews(crops, origins, move_i, ly, lx, fi.size)
        d, corr, peak, p_view = _localize_appended(
            view_idx,
            stats,
            Fs,
            Fneus,
            views,
            move_i,
            coef_xz,
            coef_yz,
            factor,
            x00,
            y00,
            depth_min,
            depth_max,
            stride_rs,
        )
        col, row, z = float(d["center"][0]), float(d["center"][1]), float(d["center"][2])
        if abs(z - float(PLANE_Z_BASE[int(m2["plane_id"])])) > TRUST_D_Z_UM:
            col, row, z = float(col0), float(row0), z_place
            p_view = int(np.nanargmax(stride_rs)) + 1
            corr = float(stride_rs[p_view - 1])
        x1m, y1m = apply_xy(np.array([col]), np.array([row]), tfm)
        d_xy = float(np.hypot(float(x1m[0]) - float(m2["x"]), float(y1m[0]) - float(m2["y"])))
        d_z = abs(z - float(m2["z_um"]))
        trusted = (
            np.isfinite(corr)
            and corr >= TRUST_R_MIN
            and d_xy <= TRUST_D_XY_PX
            and d_z <= TRUST_D_Z_UM
        )
        rec = {
            "m2_plane": key[0],
            "m2_roi": key[1],
            "view_roi_idx": {str(v): view_idx[v] for v in range(1, 5)},
            "placed_z": z_place,
            "placed_col": float(col0),
            "placed_row": float(row0),
            "promoted": bool(trusted),
            "corr": float(corr) if np.isfinite(corr) else None,
            "d_xy_px": d_xy,
            "d_z_um": d_z,
        }
        if prev is None:
            mp.setdefault("appended", []).append(rec)
        else:
            prev.update(rec)
        print(
            "    localize z=%.1f xy=(%.1f, %.1f) corr=%.3f dxy=%.2f dz=%.1f -> %s"
            % (z, col, row, corr, d_xy, d_z, "TRUSTED" if trusted else "keep unmatched"),
            flush=True,
        )
        if not trusted:
            _save_map(mp)
            gc.collect()
            continue
        n_t = min(int(tr_m2.size), int(Fs[p_view].shape[1]))
        m1_idx = _promote_files(
            m2,
            d,
            col,
            row,
            z,
            corr,
            peak,
            p_view,
            d_xy,
            d_z,
            float(x1m[0]),
            float(y1m[0]),
            np.asarray(Fs[p_view][view_idx[p_view]], dtype=np.float64)[:n_t],
            match_csv,
        )
        rec["m1_idx"] = m1_idx
        promoted_keys.add(key)
        um_rows = [
            r
            for r in um_rows
            if (int(r["m2_plane"]), int(r["m2_roi"])) not in promoted_keys
        ]
        _write_unmatched(unmatch_csv, um_rows, um_fields)
        _save_map(mp)
        gc.collect()

    print(
        "promoted %d  skipped_low_r %d  unmatched left %d  map %s"
        % (len(promoted_keys), n_skip_r, len(um_rows), MAP_PATH),
        flush=True,
    )


if __name__ == "__main__":
    main()
