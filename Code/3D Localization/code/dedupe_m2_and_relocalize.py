# -*- coding: utf-8 -*-
"""Deduplicate cross-plane m2 ROIs (keep higher SNR), then refresh 3D catalog.

Saves footprints+traces before and after dedupe for later figure updates.
Reuses existing m2_3d positions when available; re-refines only kept ROIs
that lack a valid 3D entry.
"""
from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from tplfm_utils import compute_move as _compute_move, corrcoef_scalar
from dataset_config import move_divisor_from_coef, rel_to_project

from match_linear_runner_to_recon_planes import (
    MATCH_ROOT,
    M1_OUT,
    load_matches_csv,
    load_method1,
    load_method2,
)
from recover_m2_from_bin import (
    COV_DIR,
    build_synthetic_entry,
    inverse_xy,
    load_xy_tfm,
    open_memmaps,
    resolve_match_dir,
)
from refine_z_from_footprints import (
    AVG_BIN,
    AveragedVolumes,
    REFINE_DIR,
    ab_test_from_depths,
    apply_plane_z_convention,
    refine_one,
)

DEDUPE_DIR = COV_DIR / "dedupe"
D_XY_MAX = 10.0
R_MIN = 0.60
TRUST_R_MIN = 0.40
TRUST_D_XY = 20.0
TRUST_D_Z = 12.0


def snr_trace(tr: np.ndarray) -> float:
    """Peak-ish signal over robust noise (higher = better)."""
    tr = np.asarray(tr, dtype=np.float64).ravel()
    if tr.size < 10:
        return 0.0
    med = float(np.median(tr))
    sig = float(np.percentile(tr, 95) - med)
    mad = float(np.median(np.abs(np.diff(tr))))
    noise = mad / 0.6745 if mad > 1e-12 else float(tr.std() + 1e-12)
    return sig / (noise + 1e-12)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def pack_roi(m: dict[str, Any], snr: float) -> dict[str, Any]:
    return {
        "plane_id": int(m["plane_id"]),
        "roi_idx": int(m["roi_idx"]),
        "z_um_plane": float(m["z_um"]),
        "x": float(m["x"]),
        "y": float(m["y"]),
        "xpix": np.asarray(m["xpix"], dtype=np.int32).copy(),
        "ypix": np.asarray(m["ypix"], dtype=np.int32).copy(),
        "trace": np.asarray(m["trace"], dtype=np.float64).copy(),
        "snr": float(snr),
        "plane_dir": str(m.get("plane_dir", "")),
    }


def dedupe_m2(m2_pool: list[dict[str, Any]]) -> tuple[list[int], list[dict], list[dict], list[dict]]:
    """Return (kept_indices, pair_rows, component_rows, packs_all)."""
    n = len(m2_pool)
    snrs = np.array([snr_trace(m["trace"]) for m in m2_pool], dtype=np.float64)
    xy = np.array([[m["x"], m["y"]] for m in m2_pool], dtype=np.float64)
    plane = np.array([m["plane_id"] for m in m2_pool], dtype=int)
    T = min(int(m["trace"].shape[0]) for m in m2_pool)
    st = 3
    tr = np.stack([np.asarray(m["trace"][:T:st], dtype=np.float64) for m in m2_pool], 0)

    tree = cKDTree(xy)
    uf = UnionFind(n)
    pair_rows = []
    for i, j in tree.query_pairs(float(D_XY_MAX)):
        if plane[i] == plane[j]:
            continue
        dxy = float(np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]))
        r = corrcoef_scalar(tr[i], tr[j])
        if not np.isfinite(r) or r < R_MIN or dxy > D_XY_MAX:
            continue
        uf.union(i, j)
        pair_rows.append(
            {
                "i": i,
                "j": j,
                "plane_i": int(plane[i]),
                "roi_i": int(m2_pool[i]["roi_idx"]),
                "plane_j": int(plane[j]),
                "roi_j": int(m2_pool[j]["roi_idx"]),
                "d_xy_px": dxy,
                "corr": float(r),
                "snr_i": float(snrs[i]),
                "snr_j": float(snrs[j]),
                "keep": "i" if snrs[i] >= snrs[j] else "j",
            }
        )

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(uf.find(i), []).append(i)

    kept = []
    comp_rows = []
    for root, members in comps.items():
        if len(members) == 1:
            kept.append(members[0])
            continue
        # keep max SNR
        best = max(members, key=lambda k: float(snrs[k]))
        kept.append(best)
        for m in members:
            comp_rows.append(
                {
                    "component": int(root),
                    "plane": int(plane[m]),
                    "roi": int(m2_pool[m]["roi_idx"]),
                    "snr": float(snrs[m]),
                    "kept": int(m == best),
                    "kept_plane": int(plane[best]),
                    "kept_roi": int(m2_pool[best]["roi_idx"]),
                    "n_in_component": len(members),
                }
            )

    packs_all = [pack_roi(m2_pool[i], float(snrs[i])) for i in range(n)]
    kept = sorted(kept)
    print(
        "dedupe: pairs=%d  components_multi=%d  kept=%d / %d  dropped=%d"
        % (
            len(pair_rows),
            sum(1 for ms in comps.values() if len(ms) > 1),
            len(kept),
            n,
            n - len(kept),
        )
    )
    return kept, pair_rows, comp_rows, packs_all


def save_pool(path: Path, packs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.array(packs, dtype=object))
    # also flat traces for convenience
    T = min(p["trace"].shape[0] for p in packs)
    traces = np.stack([p["trace"][:T] for p in packs], axis=0)
    meta = np.array(
        [(p["plane_id"], p["roi_idx"], p["x"], p["y"], p["z_um_plane"], p["snr"]) for p in packs],
        dtype=[
            ("plane_id", "i4"),
            ("roi_idx", "i4"),
            ("x", "f8"),
            ("y", "f8"),
            ("z_um_plane", "f8"),
            ("snr", "f8"),
        ],
    )
    np.savez_compressed(str(path.with_suffix(".npz")), traces=traces, meta=meta)
    print("wrote", path, "and", path.with_suffix(".npz"))


def main() -> None:
    DEDUPE_DIR.mkdir(parents=True, exist_ok=True)
    print("DEDUPE_DIR =", DEDUPE_DIR)

    m2_pool = load_method2()
    kept_idx, pair_rows, comp_rows, packs_all = dedupe_m2(m2_pool)
    packs_kept = [packs_all[i] for i in kept_idx]
    kept_keys = {(p["plane_id"], p["roi_idx"]) for p in packs_kept}

    # --- save before / after ---
    save_pool(DEDUPE_DIR / "m2_rois_before_dedupe.npy", packs_all)
    save_pool(DEDUPE_DIR / "m2_rois_after_dedupe.npy", packs_kept)

    with open(DEDUPE_DIR / "duplicate_pairs_snr.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(pair_rows[0].keys()) if pair_rows else [
            "i", "j", "plane_i", "roi_i", "plane_j", "roi_j", "d_xy_px", "corr", "snr_i", "snr_j", "keep"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pair_rows)

    with open(DEDUPE_DIR / "dedupe_components.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(comp_rows[0].keys()) if comp_rows else [
            "component", "plane", "roi", "snr", "kept"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(comp_rows)

    with open(DEDUPE_DIR / "kept_rois.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["plane_id", "roi_idx", "x", "y", "z_um_plane", "snr"]
        )
        w.writeheader()
        for p in packs_kept:
            w.writerow(
                {
                    "plane_id": p["plane_id"],
                    "roi_idx": p["roi_idx"],
                    "x": p["x"],
                    "y": p["y"],
                    "z_um_plane": p["z_um_plane"],
                    "snr": p["snr"],
                }
            )

    summary = {
        "n_before": len(packs_all),
        "n_after": len(packs_kept),
        "n_dropped": len(packs_all) - len(packs_kept),
        "n_pairs": len(pair_rows),
        "gates": {"D_XY_MAX": D_XY_MAX, "R_MIN": R_MIN},
        "snr_def": " (p95 - median) / (mad(diff)/0.6745)",
    }
    with open(DEDUPE_DIR / "dedupe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("dedupe summary:", summary)

    # --- rebuild 3D for kept set ---
    cat_path = COV_DIR / "m2_3d_catalog.csv"
    old_cat = {}
    if cat_path.is_file():
        for r in csv.DictReader(open(cat_path, encoding="utf-8")):
            old_cat[(int(r["m2_plane"]), int(r["m2_roi"]))] = r

    match_dir = resolve_match_dir()
    matches = load_matches_csv(match_dir / "matches.csv")
    trusted = [
        m
        for m in matches
        if float(m["corr"]) >= TRUST_R_MIN
        and float(m["d_xy_px"]) <= TRUST_D_XY
        and float(m["d_z_um"]) <= TRUST_D_Z
        and (int(m["m2_plane"]), int(m["m2_roi"])) in kept_keys
    ]
    trusted_keys = {(int(m["m2_plane"]), int(m["m2_roi"])) for m in trusted}
    print("trusted among kept:", len(trusted_keys))

    entries_m1, c1_orig, primaries = load_method1()
    tfm = load_xy_tfm(match_dir)

    # plane z from current flip flag
    from match_linear_runner_to_recon_planes import PLANE_Z_BASE, PLANE_Z_FLIPPED

    if PLANE_Z_FLIPPED:
        plane_z = {k: -v for k, v in PLANE_Z_BASE.items()}
    else:
        plane_z = dict(PLANE_Z_BASE)

    need_refine = []
    catalog_rows = []
    dict_entries = []

    for p in packs_kept:
        key = (p["plane_id"], p["roi_idx"])
        if key in trusted_keys:
            m = next(
                x
                for x in trusted
                if int(x["m2_plane"]) == key[0] and int(x["m2_roi"]) == key[1]
            )
            i1 = int(m["m1_idx"])
            col, row, z = float(c1_orig[i1, 0]), float(c1_orig[i1, 1]), float(c1_orig[i1, 2])
            catalog_rows.append(
                {
                    "m2_plane": key[0],
                    "m2_roi": key[1],
                    "source": "matched_trusted",
                    "col": col,
                    "row": row,
                    "z_um": z,
                    "corr_or_best_r": float(m["corr"]),
                    "primary_view": int(primaries[i1]),
                    "m1_idx": i1,
                    "d_xy_px": float(m["d_xy_px"]),
                    "d_z_um": float(m["d_z_um"]),
                    "n_pass": "",
                    "accepted": 1,
                    "snr": p["snr"],
                }
            )
            e = entries_m1[i1]
            dict_entries.append(
                {
                    "center": np.array([col, row, z], dtype=np.float64),
                    "center_allview": np.asarray(
                        e.get("center_allview", np.zeros((2, 4))), dtype=np.float64
                    ).copy(),
                    "pixels1": np.asarray(e.get("pixels1", []), dtype=np.float64).copy(),
                    "pixels2": np.asarray(e.get("pixels2", []), dtype=np.float64).copy(),
                    "primary_view": int(primaries[i1]),
                    "peak_corr": float(m["corr"]),
                    "source": "matched_trusted",
                    "m2_plane": key[0],
                    "m2_roi": key[1],
                    "m1_idx": i1,
                    "snr": p["snr"],
                }
            )
        else:
            # reuse previous refined/recovered if present and accepted
            old = old_cat.get(key)
            reuse = (
                old is not None
                and old.get("source") in ("recovered_bin", "recovered_bin_refined")
                and old.get("z_um") not in ("", None)
                and str(old.get("accepted", "1")) not in ("0", "false", "False")
            )
            if reuse:
                catalog_rows.append(
                    {
                        "m2_plane": key[0],
                        "m2_roi": key[1],
                        "source": "recovered_bin_refined",
                        "col": float(old["col"]),
                        "row": float(old["row"]),
                        "z_um": float(old["z_um"]),
                        "corr_or_best_r": float(old["corr_or_best_r"])
                        if old.get("corr_or_best_r") not in ("", None)
                        else "",
                        "primary_view": old.get("primary_view", ""),
                        "m1_idx": -1,
                        "d_xy_px": "",
                        "d_z_um": abs(float(old["z_um"]) - float(plane_z[key[0]])),
                        "n_pass": old.get("n_pass", ""),
                        "accepted": 1,
                        "snr": p["snr"],
                    }
                )
                # dict placeholder; filled after optional refine batch for missing only
            else:
                need_refine.append(p)

    print("reuse recovered:", sum(1 for r in catalog_rows if r["source"].startswith("recovered")))
    print("need refine:", len(need_refine))

    results = []
    if need_refine:
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
        print("\n=== average volumes for refine ===", flush=True)
        vols = AveragedVolumes(mems, AVG_BIN)
        m2_map = {(m["plane_id"], m["roi_idx"]): m for m in m2_pool}
        for k, p in enumerate(need_refine):
            m2 = m2_map[(p["plane_id"], p["roi_idx"])]
            # seed from inverse XY
            col0, row0 = inverse_xy(p["x"], p["y"], tfm)
            t0 = time.time()
            res = refine_one(
                m2, col0, row0, float(plane_z[p["plane_id"]]), tfm, vols,
                move_i, coef_xz, coef_yz, factor, x0, y0,
            )
            results.append(res)
            print(
                "  [%d/%d] p%d/roi%d z->%.1f r=%.3f %.2fs"
                % (k + 1, len(need_refine), p["plane_id"], p["roi_idx"], res["best_z"], res["best_r"], time.time() - t0),
                flush=True,
            )
        vols.release()
        del vols
        gc.collect()

        for res in results:
            key = (int(res["m2"]["plane_id"]), int(res["m2"]["roi_idx"]))
            snr = next(p["snr"] for p in packs_kept if (p["plane_id"], p["roi_idx"]) == key)
            catalog_rows.append(
                {
                    "m2_plane": key[0],
                    "m2_roi": key[1],
                    "source": "recovered_bin_refined",
                    "col": res["col"],
                    "row": res["row"],
                    "z_um": res["best_z"],
                    "corr_or_best_r": res["best_r"],
                    "primary_view": int(res["primary_v0"] + 1),
                    "m1_idx": -1,
                    "d_xy_px": "",
                    "d_z_um": abs(float(res["best_z"]) - float(plane_z[key[0]])),
                    "n_pass": res["n_pass"],
                    "accepted": 1,
                    "snr": snr,
                }
            )
            e = build_synthetic_entry(res, coef_xz, coef_yz, factor, x0, y0)
            e["source"] = "recovered_bin_refined"
            e["snr"] = snr
            e["trace_primary"] = np.asarray(res["traces"][res["primary_v0"]], dtype=np.float64)
            dict_entries.append(e)

    # For reused recovered without new dict entry, load from old dictionary if possible
    old_dict_path = COV_DIR / "m2_3d_dictionary.npy"
    old_dict_map = {}
    if old_dict_path.is_file():
        for e in np.load(str(old_dict_path), allow_pickle=True):
            if "m2_plane" in e and "m2_roi" in e:
                old_dict_map[(int(e["m2_plane"]), int(e["m2_roi"]))] = e

    have_dict = {(int(e["m2_plane"]), int(e["m2_roi"])) for e in dict_entries}
    for r in catalog_rows:
        key = (int(r["m2_plane"]), int(r["m2_roi"]))
        if key in have_dict:
            continue
        if key in old_dict_map:
            e = dict(old_dict_map[key])
            e["snr"] = r["snr"]
            dict_entries.append(e)
            have_dict.add(key)

    # A/B on kept recovered depths (optional refresh)
    recovered_for_ab = []
    for r in catalog_rows:
        if not str(r["source"]).startswith("recovered"):
            continue
        # fake res-like for ab_test
        recovered_for_ab.append(
            {
                "m2": {"plane_id": int(r["m2_plane"]), "roi_idx": int(r["m2_roi"])},
                "best_z": float(r["z_um"]),
            }
        )
    if recovered_for_ab:
        ab = ab_test_from_depths(recovered_for_ab, trusted, c1_orig)
        print("A/B after dedupe:", ab["choose"], ab)
        with open(DEDUPE_DIR / "plane_z_ab_test_after_dedupe.json", "w", encoding="utf-8") as f:
            json.dump(ab, f, indent=2)
        plane_z = apply_plane_z_convention(ab["choose"])
        # update d_z_um
        for r in catalog_rows:
            if r.get("z_um") not in ("", None) and str(r["source"]).startswith("recovered"):
                r["d_z_um"] = abs(float(r["z_um"]) - float(plane_z[int(r["m2_plane"])]))

    # sort catalog
    catalog_rows.sort(key=lambda r: (int(r["m2_plane"]), int(r["m2_roi"])))

    # archive previous catalog
    if cat_path.is_file():
        arch = DEDUPE_DIR / "m2_3d_catalog_before_dedupe.csv"
        arch.write_bytes(cat_path.read_bytes())
        print("archived previous catalog ->", arch)

    fields = list(catalog_rows[0].keys())
    with open(cat_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(catalog_rows)
    np.save(str(COV_DIR / "m2_3d_dictionary.npy"), np.array(dict_entries, dtype=object))

    cov = {
        "n_m2_before_dedupe": len(packs_all),
        "n_m2_after_dedupe": len(packs_kept),
        "n_with_3d": sum(1 for r in catalog_rows if r.get("z_um") not in ("", None)),
        "by_source": {},
        "dedupe_dir": rel_to_project(DEDUPE_DIR),
    }
    for r in catalog_rows:
        s = r["source"]
        cov["by_source"][s] = cov["by_source"].get(s, 0) + 1
    with open(COV_DIR / "coverage_summary.json", "w", encoding="utf-8") as f:
        json.dump(cov, f, indent=2)
    print("coverage:", cov)
    print("\nDone.")


if __name__ == "__main__":
    main()
