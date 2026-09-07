# -*- coding: utf-8 -*-
"""Refine recovered m2 3D positions (block-averaged search).

1. Stream-average each view data.bin (bin=10), never hold full T in RAM.
2. Depth + footprint search on averaged volumes (fast, better SNR than stride).
3. Plane z-sign A/B (global flip of PLANE_Z_BASE) is run LAST from refined depths.
"""
from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tplfm_utils import compute_move as _compute_move, corrcoef_scalar
from dataset_config import move_divisor_from_coef

from match_linear_runner_to_recon_planes import (
    MATCH_ROOT,
    M1_OUT,
    PLANE_Z_BASE,
    load_matches_csv,
    load_method1,
    load_method2,
)
from recover_m2_from_bin import (
    COV_DIR,
    build_synthetic_entry,
    footprint_at_center,
    inverse_xy,
    load_xy_tfm,
    open_memmaps,
    resolve_match_dir,
    view_center_in_bin,
    warped_m2_footprint,
)

REFINE_DIR = COV_DIR / "z_refined"
Z_MIN, Z_MAX, Z_STEP = -40.0, 40.0, 2.0
Z_REFINE_HALF = 20.0
XY_LOCAL = 8
AVG_BIN = 10  # 6000 / 10 = 600 frames
R_PASS = 0.12
TRUST_R_MIN = 0.40
TRUST_D_XY = 20.0
TRUST_D_Z = 12.0


class AveragedVolumes:
    """vols[v]: (n_avg, Ly, Lx) float32 from block-mean of data.bin."""

    def __init__(self, mems: list, avg_bin: int = AVG_BIN) -> None:
        self.avg_bin = int(avg_bin)
        self.vols: list[np.ndarray] = []
        nf = int(mems[0].shape[0])
        self.n_avg = nf // self.avg_bin
        self.ly = int(mems[0].shape[1])
        self.lx = int(mems[0].shape[2])
        t0 = time.time()
        for v, mem in enumerate(mems):
            print(
                "  average view%d: T=%d -> %d (bin=%d) ..."
                % (v + 1, nf, self.n_avg, self.avg_bin),
                flush=True,
            )
            out = np.empty((self.n_avg, self.ly, self.lx), dtype=np.float32)
            # stream one bin at a time — peak RAM ~ bin * FOV
            for i in range(self.n_avg):
                s = i * self.avg_bin
                e = s + self.avg_bin
                block = np.asarray(mem[s:e], dtype=np.float32)
                out[i] = block.mean(axis=0)
                del block
            self.vols.append(out)
            print(
                "    done view%d  elapsed=%.1fs  vol=%.2fGB"
                % (v + 1, time.time() - t0, out.nbytes / 1e9),
                flush=True,
            )
            gc.collect()
        print(
            "average preload done %.1fs  total RAM~%.2f GB"
            % (time.time() - t0, sum(a.nbytes for a in self.vols) / 1e9),
            flush=True,
        )

    def trace(self, v: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        n = self.n_avg
        if xs.size == 0:
            return np.zeros(n, dtype=np.float64)
        xs = np.asarray(xs, dtype=np.int64)
        ys = np.asarray(ys, dtype=np.int64)
        ok = (xs >= 0) & (xs < self.lx) & (ys >= 0) & (ys < self.ly)
        xs, ys = xs[ok], ys[ok]
        if xs.size < 3:
            return np.zeros(n, dtype=np.float64)
        if xs.size > 64:
            sel = np.linspace(0, xs.size - 1, 64).astype(int)
            xs, ys = xs[sel], ys[sel]
        return self.vols[v][:, ys, xs].mean(axis=1).astype(np.float64)

    def release(self) -> None:
        self.vols.clear()
        gc.collect()


def block_average_trace(tr: np.ndarray, avg_bin: int, n_avg: int) -> np.ndarray:
    tr = np.asarray(tr, dtype=np.float64).ravel()
    need = n_avg * avg_bin
    if tr.size < need:
        pad = np.zeros(need, dtype=np.float64)
        pad[: tr.size] = tr
        tr = pad
    else:
        tr = tr[:need]
    return tr.reshape(n_avg, avg_bin).mean(axis=1)


def eval_views_at(
    col, row, z, p0, vols, tr_m2, move_i, coef_xz, coef_yz, factor, x0, y0, seed_fp
):
    rs = np.full(4, np.nan)
    traces = np.zeros((4, vols.n_avg), dtype=np.float64)
    seed_xs = seed_fp[0] if seed_fp else None
    seed_ys = seed_fp[1] if seed_fp else None
    for v0 in range(4):
        cx, cy = view_center_in_bin(
            col, row, z, p0, v0, move_i, coef_xz, coef_yz, factor, x0, y0
        )
        xs, ys = footprint_at_center(seed_xs, seed_ys, cx, cy, vols.ly, vols.lx)
        traces[v0] = vols.trace(v0, xs, ys)
        r = corrcoef_scalar(traces[v0], tr_m2)
        if np.isfinite(r):
            rs[v0] = r
    rmax = float(np.nanmax(rs)) if np.any(np.isfinite(rs)) else -np.inf
    score = float(np.sum(np.clip(np.nan_to_num(rs, nan=0.0), 0.0, None)))
    return rs, traces, rmax, score


def find_view_footprints(
    col0, row0, z_hint, p0, vols, tr_m2, move_i, coef_xz, coef_yz, factor, x0, y0, seed_fp
):
    seed_xs = seed_fp[0] if seed_fp else None
    seed_ys = seed_fp[1] if seed_fp else None
    cx_out = np.full(4, np.nan)
    cy_out = np.full(4, np.nan)
    rs = np.full(4, np.nan)
    for v0 in range(4):
        cx0, cy0 = view_center_in_bin(
            col0, row0, z_hint, p0, v0, move_i, coef_xz, coef_yz, factor, x0, y0
        )
        best_r, best_cx, best_cy = -np.inf, cx0, cy0
        for dcol in range(-XY_LOCAL, XY_LOCAL + 1, 2):
            for drow in range(-XY_LOCAL, XY_LOCAL + 1, 2):
                cx, cy = cx0 + dcol, cy0 + drow
                xs, ys = footprint_at_center(seed_xs, seed_ys, cx, cy, vols.ly, vols.lx)
                r = corrcoef_scalar(vols.trace(v0, xs, ys), tr_m2)
                if np.isfinite(r) and r > best_r:
                    best_r, best_cx, best_cy = float(r), float(cx), float(cy)
        cx0, cy0 = best_cx, best_cy
        for dcol in (-1, 0, 1):
            for drow in (-1, 0, 1):
                if dcol == 0 and drow == 0:
                    continue
                cx, cy = cx0 + dcol, cy0 + drow
                xs, ys = footprint_at_center(seed_xs, seed_ys, cx, cy, vols.ly, vols.lx)
                r = corrcoef_scalar(vols.trace(v0, xs, ys), tr_m2)
                if np.isfinite(r) and r > best_r:
                    best_r, best_cx, best_cy = float(r), float(cx), float(cy)
        if best_r >= R_PASS:
            cx_out[v0], cy_out[v0], rs[v0] = best_cx, best_cy, best_r
    return cx_out, cy_out, rs


def fit_z_parallax(cx, cy, p0, move_i, coef_xz, coef_yz, factor, x0, y0, col0, row0, z_lo=None, z_hi=None):
    valid = np.isfinite(cx) & np.isfinite(cy)
    if int(valid.sum()) < 2:
        return float("nan"), col0, row0, float("nan")
    if z_lo is None:
        z_lo = Z_MIN
    if z_hi is None:
        z_hi = Z_MAX
    best = (float("nan"), col0, row0, np.inf)
    for z in np.arange(z_lo, z_hi + Z_STEP / 2, Z_STEP):
        for dcol in (-2, 0, 2):
            for drow in (-2, 0, 2):
                col, row = col0 + dcol, row0 + drow
                err2, n = 0.0, 0
                for v0 in range(4):
                    if not valid[v0]:
                        continue
                    px, py = view_center_in_bin(
                        col, row, float(z), p0, v0, move_i, coef_xz, coef_yz, factor, x0, y0
                    )
                    err2 += (px - cx[v0]) ** 2 + (py - cy[v0]) ** 2
                    n += 1
                rms = float(np.sqrt(err2 / max(n, 1)))
                if rms < best[3]:
                    best = (float(z), float(col), float(row), rms)
    return best


def refine_one(m2, col_seed, row_seed, z_old, tfm, vols, move_i, coef_xz, coef_yz, factor, x0, y0):
    tr_m2 = block_average_trace(m2["trace"], vols.avg_bin, vols.n_avg)
    seed_fp = warped_m2_footprint(m2, tfm, vols.ly, vols.lx)
    args = (vols, tr_m2, move_i, coef_xz, coef_yz, factor, x0, y0, seed_fp)

    z_seed = float(z_old) if np.isfinite(z_old) else float(m2["z_um"])
    z_lo = max(Z_MIN, z_seed - Z_REFINE_HALF)
    z_hi = min(Z_MAX, z_seed + Z_REFINE_HALF)

    best_p0, best_score = 0, -np.inf
    for p0 in range(4):
        _rs, _tr, _rmax, score = eval_views_at(col_seed, row_seed, z_seed, p0, *args)
        if score > best_score:
            best_p0, best_score = p0, score
    p0 = int(best_p0)

    depth_curve = []
    best_z_act, best_sc = z_seed, -np.inf
    for z in np.arange(z_lo, z_hi + Z_STEP / 2, Z_STEP):
        _rs, _tr, _rmax, score = eval_views_at(col_seed, row_seed, float(z), p0, *args)
        depth_curve.append((float(z), float(score)))
        if score > best_sc:
            best_sc, best_z_act = score, float(z)

    best_col, best_row = float(col_seed), float(row_seed)
    best_xy = best_sc
    for dcol in range(-6, 7, 2):
        for drow in range(-6, 7, 2):
            if dcol == 0 and drow == 0:
                continue
            _rs, _tr, _rmax, score = eval_views_at(
                col_seed + dcol, row_seed + drow, best_z_act, p0, *args
            )
            if score > best_xy:
                best_xy = score
                best_col, best_row = float(col_seed + dcol), float(row_seed + drow)

    cx, cy, rs_fp = find_view_footprints(
        best_col, best_row, best_z_act, p0, vols, tr_m2, move_i, coef_xz, coef_yz, factor, x0, y0, seed_fp
    )
    z_geo, col_geo, row_geo, rms = fit_z_parallax(
        cx, cy, p0, move_i, coef_xz, coef_yz, factor, x0, y0, best_col, best_row,
        z_lo=z_lo, z_hi=z_hi,
    )
    if np.isfinite(z_geo) and np.isfinite(rms) and rms <= 6.0:
        best_z, best_col, best_row, z_source = z_geo, col_geo, row_geo, "parallax"
    else:
        best_z, z_source = best_z_act, "activity"

    rs, traces_avg, rmax, score = eval_views_at(best_col, best_row, best_z, p0, *args)
    n_pass = int(np.sum(np.isfinite(rs) & (rs >= R_PASS)))

    # footprint deltas at primary for later full-T extract
    seed_xs = seed_fp[0] if seed_fp else None
    seed_ys = seed_fp[1] if seed_fp else None
    cx_p, cy_p = view_center_in_bin(
        best_col, best_row, best_z, p0, p0, move_i, coef_xz, coef_yz, factor, x0, y0
    )
    xs_p, ys_p = footprint_at_center(seed_xs, seed_ys, cx_p, cy_p, vols.ly, vols.lx)

    return {
        "m2": m2,
        "col": float(best_col),
        "row": float(best_row),
        "z2": float(m2["z_um"]),
        "z_old": float(z_old),
        "best_z": float(best_z),
        "z_activity": float(best_z_act),
        "z_parallax": float(z_geo) if np.isfinite(z_geo) else float("nan"),
        "parallax_rms_px": float(rms) if np.isfinite(rms) else float("nan"),
        "z_source": z_source,
        "best_r": float(rmax),
        "primary_v0": p0,
        "rs": rs,
        "n_pass": n_pass,
        "accepted": True,
        "traces": traces_avg,  # placeholder; replaced with full-T later
        "centers_x": cx,
        "centers_y": cy,
        "fp_rs": rs_fp,
        "depth_curve": depth_curve,
        "fp_xs": xs_p,
        "fp_ys": ys_p,
        "fp_view": p0,
    }


def extract_full_traces(results: list[dict], mems: list) -> None:
    """Replace averaged traces with full-T primary-view traces from memmaps."""
    print("\n=== extract full-T traces from original bins ===", flush=True)
    nf = int(mems[0].shape[0])
    chunk = 200
    for k, r in enumerate(results):
        v0 = int(r["fp_view"])
        xs = np.asarray(r["fp_xs"], dtype=np.int64)
        ys = np.asarray(r["fp_ys"], dtype=np.int64)
        ly, lx = mems[v0].shape[1], mems[v0].shape[2]
        ok = (xs >= 0) & (xs < lx) & (ys >= 0) & (ys < ly)
        xs, ys = xs[ok], ys[ok]
        if xs.size == 0:
            r["traces"] = np.zeros((4, nf), dtype=np.float64)
            r["traces"][v0] = 0
            continue
        if xs.size > 64:
            sel = np.linspace(0, xs.size - 1, 64).astype(int)
            xs, ys = xs[sel], ys[sel]
        out = np.zeros(nf, dtype=np.float64)
        mem = mems[v0]
        for s in range(0, nf, chunk):
            e = min(s + chunk, nf)
            # read span then mean footprint — one contiguous read
            block = np.asarray(mem[s:e], dtype=np.float32)
            out[s:e] = block[:, ys, xs].mean(axis=1)
            del block
        traces = np.zeros((4, nf), dtype=np.float64)
        traces[v0] = out
        r["traces"] = traces
        if (k + 1) % 25 == 0 or k == 0 or k + 1 == len(results):
            print("  full-T traces %d/%d" % (k + 1, len(results)), flush=True)
        if (k + 1) % 10 == 0:
            gc.collect()
    gc.collect()


def ab_test_from_depths(
    results: list[dict],
    trusted: list[dict],
    c1_orig: np.ndarray,
) -> dict[str, Any]:
    """A/B after all depths known: PLANE_Z_BASE vs global sign flip."""

    def score(flipped: bool) -> dict[str, float]:
        plane = {k: (-v if flipped else v) for k, v in PLANE_Z_BASE.items()}
        by_p: dict[int, list[float]] = {pid: [] for pid in plane}
        for r in results:
            pid = int(r["m2"]["plane_id"])
            by_p.setdefault(pid, []).append(float(r["best_z"]))
        rec = {}
        trust = {}
        vals = []
        for pid in plane:
            zs = np.array(by_p.get(pid, []), dtype=float)
            rec[pid] = (
                float(np.median(np.abs(zs - plane[pid]))) if zs.size else float("nan")
            )
            zs_t = []
            for m in trusted:
                if int(m["m2_plane"]) != pid:
                    continue
                i1 = int(m["m1_idx"])
                if 0 <= i1 < len(c1_orig):
                    zs_t.append(float(c1_orig[i1, 2]))
            zs_a = np.array(zs_t, dtype=float)
            trust[pid] = (
                float(np.median(np.abs(zs_a - plane[pid]))) if zs_a.size else float("nan")
            )
            vals.extend([rec[pid], trust[pid]])
        out: dict[str, float] = {"score": float(np.nanmean(vals))}
        for pid in plane:
            out["rec_med_abs_%d" % pid] = rec[pid]
            out["trust_med_abs_%d" % pid] = trust[pid]
            out["rec_median_z_%d" % pid] = (
                float(np.median(by_p[pid])) if by_p[pid] else float("nan")
            )
        return out

    A = score(False)
    B = score(True)
    choose = "B" if B["score"] < A["score"] else "A"
    return {"A": A, "B": B, "choose": choose}


def apply_plane_z_convention(choose: str) -> dict[int, float]:
    path = Path(__file__).resolve().parent / "match_linear_runner_to_recon_planes.py"
    text = path.read_text(encoding="utf-8")
    if "PLANE_Z_FLIPPED" not in text:
        raise RuntimeError("PLANE_Z_FLIPPED missing in match script")
    if choose == "B":
        text = text.replace("PLANE_Z_FLIPPED = False", "PLANE_Z_FLIPPED = True")
        plane = {k: -v for k, v in PLANE_Z_BASE.items()}
    else:
        text = text.replace("PLANE_Z_FLIPPED = True", "PLANE_Z_FLIPPED = False")
        plane = dict(PLANE_Z_BASE)
    path.write_text(text, encoding="utf-8")
    print("PLANE_Z_FLIPPED ->", choose == "B", "plane_z", plane)
    return plane


def main() -> None:
    REFINE_DIR.mkdir(parents=True, exist_ok=True)
    cat_path = COV_DIR / "m2_3d_catalog.csv"
    rows = list(csv.DictReader(open(cat_path, encoding="utf-8")))

    m2_pool = load_method2()
    m2_map = {(m["plane_id"], m["roi_idx"]): i for i, m in enumerate(m2_pool)}
    match_dir = resolve_match_dir()
    tfm = load_xy_tfm(match_dir)
    matches = load_matches_csv(match_dir / "matches.csv")
    trusted = [
        m
        for m in matches
        if float(m["corr"]) >= TRUST_R_MIN
        and float(m["d_xy_px"]) <= TRUST_D_XY
        and float(m["d_z_um"]) <= TRUST_D_Z
    ]
    trusted_keys = {(int(m["m2_plane"]), int(m["m2_roi"])) for m in trusted}
    to_refine = [
        r
        for r in rows
        if r.get("source") in ("recovered_bin", "recovered_bin_refined")
        and (int(r["m2_plane"]), int(r["m2_roi"])) not in trusted_keys
        and r.get("z_um") not in ("", None)
    ]
    print(
        "refine candidates: %d  trusted skipped: %d  AVG_BIN=%d"
        % (len(to_refine), len(trusted_keys), AVG_BIN),
    )
    if not to_refine:
        print("nothing to refine; catalog left as recover wrote it")
        with open(REFINE_DIR / "refine_summary.json", "w", encoding="utf-8") as f:
            json.dump({"n_refined": 0, "trusted_skipped": len(trusted_keys)}, f, indent=2)
        return

    _entries, c1_orig, _prim = load_method1()

    coef = np.load(str(M1_OUT / "psf_fit_coef.npz"))
    coef_xz = np.asarray(coef["coef_psf_xz"], dtype=np.float64)
    coef_yz = np.asarray(coef["coef_psf_yz"], dtype=np.float64)
    factor = float(coef["factor"])
    x0, y0 = float(coef["x0"]), float(coef["y0"])
    move_div = move_divisor_from_coef(coef)
    move_i = np.round(
        _compute_move(
            np.asarray(coef["centers_1based"], dtype=np.float64), factor=move_div
        )
    ).astype(int)

    mems, ly, lx, nf = open_memmaps()
    print("\n=== block-average volumes (then free full-frame reads) ===", flush=True)
    vols = AveragedVolumes(mems, AVG_BIN)
    # memmaps stay open for later full-T extract; averaged vols are the working set

    print("\n=== refine on averaged volumes ===", flush=True)
    results = []
    t_all = time.time()
    for k, row in enumerate(to_refine):
        key = (int(row["m2_plane"]), int(row["m2_roi"]))
        m2 = m2_pool[m2_map[key]]
        t0 = time.time()
        res = refine_one(
            m2,
            float(row["col"]),
            float(row["row"]),
            float(row["z_um"]),
            tfm,
            vols,
            move_i,
            coef_xz,
            coef_yz,
            factor,
            x0,
            y0,
        )
        results.append(res)
        print(
            "  [%d/%d] p%d/roi%d  z %.1f -> %.1f (%s)  r=%.3f  %.2fs"
            % (
                k + 1,
                len(to_refine),
                key[0],
                key[1],
                res["z_old"],
                res["best_z"],
                res["z_source"],
                res["best_r"],
                time.time() - t0,
            ),
            flush=True,
        )
    print("refine elapsed %.1fs" % (time.time() - t_all), flush=True)

    print("release averaged volumes ...", flush=True)
    vols.release()
    del vols
    gc.collect()

    # A/B LAST
    print("\n=== A/B plane z sign (after all depths) ===", flush=True)
    ab = ab_test_from_depths(results, trusted, c1_orig)
    print("A (nominal):", ab["A"], flush=True)
    print("B (flipped):", ab["B"], flush=True)
    print("choose", ab["choose"], flush=True)
    with open(REFINE_DIR / "plane_z_ab_test.json", "w", encoding="utf-8") as f:
        json.dump(ab, f, indent=2)
    plane_z = apply_plane_z_convention(ab["choose"])

    # write report + catalog
    fields = [
        "m2_plane", "m2_roi", "col", "row", "z_old", "best_z", "z_activity",
        "z_parallax", "z_source", "parallax_rms_px", "best_r", "n_pass",
        "primary_view", "r_v1", "r_v2", "r_v3", "r_v4", "plane_z_prior",
    ]
    with open(REFINE_DIR / "refine_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            rs = r["rs"]
            pid = int(r["m2"]["plane_id"])
            w.writerow(
                {
                    "m2_plane": pid,
                    "m2_roi": int(r["m2"]["roi_idx"]),
                    "col": r["col"],
                    "row": r["row"],
                    "z_old": r["z_old"],
                    "best_z": r["best_z"],
                    "z_activity": r["z_activity"],
                    "z_parallax": r["z_parallax"],
                    "z_source": r["z_source"],
                    "parallax_rms_px": r["parallax_rms_px"],
                    "best_r": r["best_r"],
                    "n_pass": r["n_pass"],
                    "primary_view": int(r["primary_v0"] + 1),
                    "r_v1": rs[0],
                    "r_v2": rs[1],
                    "r_v3": rs[2],
                    "r_v4": rs[3],
                    "plane_z_prior": plane_z[pid],
                }
            )

    refine_map = {(int(r["m2"]["plane_id"]), int(r["m2"]["roi_idx"])): r for r in results}
    new_rows = []
    for row in rows:
        row = dict(row)
        key = (int(row["m2_plane"]), int(row["m2_roi"]))
        if key in refine_map:
            r = refine_map[key]
            row["col"] = r["col"]
            row["row"] = r["row"]
            row["z_um"] = r["best_z"]
            row["corr_or_best_r"] = r["best_r"]
            row["primary_view"] = int(r["primary_v0"] + 1)
            row["source"] = "recovered_bin_refined"
            row["d_z_um"] = abs(float(r["best_z"]) - float(plane_z[key[0]]))
            row["n_pass"] = r["n_pass"]
        new_rows.append(row)
    with open(cat_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
        w.writeheader()
        w.writerows(new_rows)

    dict_path = COV_DIR / "m2_3d_dictionary.npy"
    old_dict = list(np.load(str(dict_path), allow_pickle=True))
    kept = [e for e in old_dict if e.get("source") == "matched_trusted"]
    recovered_entries = [
        build_synthetic_entry(r, coef_xz, coef_yz, factor, x0, y0) for r in results
    ]
    for e, r in zip(recovered_entries, results):
        e["source"] = "recovered_bin_refined"
        e["z_source"] = r["z_source"]
        e["z_old"] = r["z_old"]
        e["trace_primary"] = np.asarray(r["traces"][r["primary_v0"]], dtype=np.float64)
    np.save(str(dict_path), np.array(kept + recovered_entries, dtype=object))

    z_new = np.array([r["best_z"] for r in results])
    plane_med = {
        ("p%d_median_z" % pid): float(
            np.median(
                [r["best_z"] for r in results if int(r["m2"]["plane_id"]) == pid] or [np.nan]
            )
        )
        for pid in PLANE_Z_BASE
    }
    summary = {
        "n_refined": len(results),
        "avg_bin": AVG_BIN,
        "plane_z_convention": ab["choose"],
        "plane_z": plane_z,
        "ab_test": ab,
        "median_z": float(np.median(z_new)),
        "z_min": float(z_new.min()),
        "z_max": float(z_new.max()),
        **plane_med,
    }
    with open(REFINE_DIR / "refine_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("summary", summary)

    readme = MATCH_ROOT / "README.md"
    if readme.is_file():
        note = (
            "\n## Plane depth convention\n\n"
            "A/B test (after full refine) selected **%s**: plane_z=%s. "
            "See `m2_3d_coverage/z_refined/plane_z_ab_test.json`.\n"
            % (ab["choose"], plane_z)
        )
        txt = readme.read_text(encoding="utf-8")
        if "## Plane depth convention" in txt:
            import re

            txt = re.sub(
                r"\n## Plane depth convention\n.*?(?=\n## |\Z)",
                note + "\n",
                txt,
                flags=re.S,
            )
        else:
            txt = txt.rstrip() + "\n" + note
        readme.write_text(txt, encoding="utf-8")

    print("\nDone.")


if __name__ == "__main__":
    main()
