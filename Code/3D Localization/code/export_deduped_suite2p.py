# -*- coding: utf-8 -*-
"""Export deduped m2 ROIs as suite2p-readable plane0 folders.

Does not modify original suite2p folders. Writes under CA1R1f5Output:

  m2_suite2p_after_dedupe/
    Plane{14,19,24,29}/suite2p/plane0/
      F.npy, Fneu.npy, spks.npy, stat.npy, iscell.npy, ops.npy
    roi_index_map.csv
    README.txt
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dataset_config import OUT_DIR, rel_to_project
from match_linear_runner_to_recon_planes import PLANE_SPECS, PLANE_Z_BASE

KEPT_CSV = OUT_DIR / "match" / "m2_3d_coverage" / "dedupe" / "kept_rois.csv"
OUT_ROOT = OUT_DIR / "m2_suite2p_after_dedupe"


def load_kept() -> dict[int, list[tuple[int, float]]]:
    """plane_id -> [(old_roi_idx, snr), ...] sorted by old_roi_idx."""
    by: dict[int, list[tuple[int, float]]] = {pid: [] for pid in PLANE_Z_BASE}
    with open(KEPT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = int(row["plane_id"])
            by.setdefault(pid, []).append((int(row["roi_idx"]), float(row["snr"])))
    for pid in by:
        by[pid].sort(key=lambda t: t[0])
    return by


def export_plane(
    plane_dir: Path, plane_id: int, kept: list[tuple[int, float]], out_plane0: Path
) -> list[dict]:
    src = plane_dir
    F = np.load(str(src / "F.npy"))
    Fneu = np.load(str(src / "Fneu.npy"))
    spks = np.load(str(src / "spks.npy"))
    stat = list(np.load(str(src / "stat.npy"), allow_pickle=True))
    iscell_src = np.load(str(src / "iscell.npy"))
    ops = np.load(str(src / "ops.npy"), allow_pickle=True).item()

    idxs = [i for i, _ in kept]
    if not idxs:
        print("plane %d: no kept ROIs, skip" % plane_id)
        return []
    for i in idxs:
        if i < 0 or i >= len(stat):
            raise IndexError("plane %d roi %d out of range" % (plane_id, i))

    F2 = np.asarray(F[idxs], dtype=np.float32)
    Fneu2 = np.asarray(Fneu[idxs], dtype=np.float32)
    spks2 = np.asarray(spks[idxs], dtype=np.float32)
    stat2 = [dict(stat[i]) for i in idxs]
    if iscell_src.ndim == 2:
        iscell2 = np.ones((len(idxs), 2), dtype=np.float64)
        iscell2[:, 1] = 1.0
    else:
        iscell2 = np.ones(len(idxs), dtype=np.float64)

    ops2 = dict(ops)
    ops2["nrois"] = int(len(idxs))

    out_plane0.mkdir(parents=True, exist_ok=True)
    np.save(str(out_plane0 / "F.npy"), F2)
    np.save(str(out_plane0 / "Fneu.npy"), Fneu2)
    np.save(str(out_plane0 / "spks.npy"), spks2)
    np.save(str(out_plane0 / "stat.npy"), np.array(stat2, dtype=object))
    np.save(str(out_plane0 / "iscell.npy"), iscell2)
    np.save(str(out_plane0 / "ops.npy"), ops2)

    rows = []
    for new_i, (old_i, snr) in enumerate(kept):
        rows.append(
            {
                "plane_id": plane_id,
                "new_roi_idx": new_i,
                "old_roi_idx": old_i,
                "snr": snr,
                "out_dir": rel_to_project(out_plane0),
            }
        )
    print(
        "plane %d: kept %d / orig %d -> %s"
        % (plane_id, len(idxs), len(stat), out_plane0)
    )
    return rows


def main() -> None:
    if not KEPT_CSV.is_file():
        raise FileNotFoundError(KEPT_CSV)
    kept_by = load_kept()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    map_rows: list[dict] = []
    for plane_dir, plane_id, _z in PLANE_SPECS:
        plane_name = plane_dir.parents[1].name  # Plane14 etc.
        out_plane0 = OUT_ROOT / plane_name / "suite2p" / "plane0"
        map_rows.extend(
            export_plane(plane_dir, int(plane_id), kept_by.get(int(plane_id), []), out_plane0)
        )

    map_csv = OUT_ROOT / "roi_index_map.csv"
    with open(map_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["plane_id", "new_roi_idx", "old_roi_idx", "snr", "out_dir"]
        )
        w.writeheader()
        w.writerows(map_rows)

    layout_lines = ["  %s/suite2p/plane0/" % pdir.parents[1].name for pdir, _pid, _z in PLANE_SPECS]
    readme = OUT_ROOT / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Deduped method2 suite2p exports (cross-plane duplicates removed; higher SNR kept).",
                "Original suite2p folders were NOT modified.",
                "",
                "Layout:",
                *layout_lines,
                "",
                "Each plane0 contains: F.npy, Fneu.npy, spks.npy, stat.npy, iscell.npy, ops.npy",
                "(no data.bin — use the original plane's data.bin / meanImg in ops if needed).",
                "",
                "ROI indices are reindexed 0..N-1 per plane; see roi_index_map.csv for old_roi_idx.",
                "Source kept list: %s" % rel_to_project(KEPT_CSV),
                "Total kept ROIs: %d" % len(map_rows),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("wrote", map_csv)
    print("OUT_ROOT =", OUT_ROOT)
    print("total kept", len(map_rows))


if __name__ == "__main__":
    main()
