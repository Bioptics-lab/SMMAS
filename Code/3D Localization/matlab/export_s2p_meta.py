# -*- coding: utf-8 -*-
"""Write slim numeric sidecars so MATLAB can load pickled suite2p stat/ops."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from dataset_config import PLANE_Z_BASE, plane_suite2p_dir, view_plane_dir  # noqa: E402
from tplfm_utils import load_npy  # noqa: E402


def _concat_pix(stat, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks = [np.asarray(st[key], dtype=np.float64).ravel() for st in stat]
    counts = np.array([c.size for c in chunks], dtype=np.int32)
    offsets = np.zeros(len(chunks), dtype=np.int32)
    if len(chunks):
        offsets[1:] = np.cumsum(counts[:-1])
    if chunks:
        cat = np.concatenate(chunks)
    else:
        cat = np.zeros((0,), dtype=np.float64)
    return cat, offsets, counts


def export_plane(plane: Path) -> Path:
    out = plane / "s2p_matlab"
    out.mkdir(exist_ok=True)
    ops = load_npy(plane / "ops.npy", allow_pickle=True).item()
    stat = list(load_npy(plane / "stat.npy", allow_pickle=True))
    xpix, xoff, xcnt = _concat_pix(stat, "xpix")
    ypix, yoff, ycnt = _concat_pix(stat, "ypix")
    if all("lam" in st for st in stat):
        lam, loff, lcnt = _concat_pix(stat, "lam")
    else:
        lam = np.ones_like(xpix)
        loff, lcnt = xoff, xcnt
    med = np.full((len(stat), 2), np.nan, dtype=np.float64)
    for i, st in enumerate(stat):
        if "med" in st:
            m = np.asarray(st["med"], dtype=np.float64).ravel()
            if m.size >= 2:
                med[i, :] = m[:2]
    np.save(out / "xpix.npy", xpix)
    np.save(out / "ypix.npy", ypix)
    np.save(out / "lam.npy", lam)
    np.save(out / "pix_offset.npy", xoff)
    np.save(out / "pix_count.npy", xcnt)
    np.save(out / "med.npy", med)
    np.save(out / "meanImg.npy", np.asarray(ops["meanImg"], dtype=np.float64))
    payload = {
        "Ly": int(ops["Ly"]),
        "Lx": int(ops["Lx"]),
        "nframes": int(ops["nframes"]),
        "nroi": len(stat),
    }
    if ops.get("yoff") is not None:
        np.save(out / "yoff.npy", np.asarray(ops["yoff"], dtype=np.float64).ravel())
        payload["has_yoff"] = True
    if ops.get("xoff") is not None:
        np.save(out / "xoff.npy", np.asarray(ops["xoff"], dtype=np.float64).ravel())
        payload["has_xoff"] = True
    (out / "ops.json").write_text(json.dumps(payload), encoding="utf-8")
    print("wrote", out, "nROI=%d FOV=%dx%d T=%d" % (len(stat), ops["Ly"], ops["Lx"], ops["nframes"]))
    return out


def main() -> None:
    for v in range(1, 5):
        export_plane(view_plane_dir(v))
    for pid in sorted(PLANE_Z_BASE):
        export_plane(plane_suite2p_dir(pid))
    print("Done.")


if __name__ == "__main__":
    main()
