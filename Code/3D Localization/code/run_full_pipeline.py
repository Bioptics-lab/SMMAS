# -*- coding: utf-8 -*-
"""Run the TPLFM 4-view localization pipeline.

TIFF->bin is low-memory and runs first (skips non-empty data.bin). Localization
and later steps wait until available physical RAM is high enough
(see wait_for_ram.py). On RAM timeout the script exits after bins, without
starting localize.

After match, unmatched recon ROIs are written into the four view suite2p
folders and rematched (promote_unmatched_via_view_suite2p.py) before recover.
Refine then only touches remaining recovered_bin rows and does not restore a
stale catalog backup over trusted matches.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dataset_config import CODE_DIR, PROJECT
from wait_for_ram import avail_gb, wait_for_ram

PY = sys.executable

# Low-memory: page-wise TIFF -> data.bin
BIN_STEP = "prepare_view_bins_from_tiff.py"

# Memory-heavy and downstream
HEAVY_STEPS = [
    "run_lr_localize.py",
    "match_linear_runner_to_recon_planes.py",
    "promote_unmatched_via_view_suite2p.py",
    "recover_m2_from_bin.py",
    "refine_z_from_footprints.py",
    "dedupe_m2_and_relocalize.py",
    "export_deduped_suite2p.py",
    "plot_m2_3d_distribution.py",
]


def run_step(name: str) -> None:
    script = CODE_DIR / name
    if not script.is_file():
        raise FileNotFoundError(script)
    print("\n======== %s  (avail RAM=%.2f GB) ========" % (name, avail_gb()), flush=True)
    r = subprocess.run([PY, str(script)], cwd=str(CODE_DIR))
    if r.returncode != 0:
        raise SystemExit("step failed: %s (exit %d)" % (name, r.returncode))


def main() -> None:
    print("PROJECT =", PROJECT)
    print("python =", PY)
    print("avail RAM now = %.2f GB" % avail_gb(), flush=True)

    run_step(BIN_STEP)

    if not wait_for_ram(poll_s=30.0, stable_hits=2, timeout_s=3.0 * 3600.0):
        print(
            "Bins are written. Localization was NOT started (RAM still low).\n"
            "Re-run this script later, or from the package root:\n"
            "  python code/run_lr_localize.py",
            flush=True,
        )
        raise SystemExit(2)

    for name in HEAVY_STEPS:
        run_step(name)
    print("\nAll pipeline steps finished.", flush=True)


if __name__ == "__main__":
    main()
