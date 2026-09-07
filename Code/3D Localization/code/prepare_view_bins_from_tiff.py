# -*- coding: utf-8 -*-
"""Convert Views/{v}/file_{v}.tif -> suite2p/plane0/data.bin (int16).

Streams one TIFF page at a time (peak RAM ~ one frame). Applies ops yoff/xoff
rigid shifts when present. Skips a view if data.bin already has the expected size
and is not all-zero.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import tifffile

from dataset_config import DATA_ROOT, view_plane_dir, view_tiff_path
from tplfm_utils import load_npy

N_VIEWS = 4


def _frame_to_int16(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint16:
        return np.clip(frame.astype(np.int32), 0, 32767).astype(np.int16)
    clipped = np.asarray(frame, dtype=np.float32)
    return np.clip(np.rint(clipped), -32768, 32767).astype(np.int16)


def _bin_ok(bin_path: Path, nf: int, ly: int, lx: int) -> bool:
    expect = nf * ly * lx * 2
    if not bin_path.is_file() or bin_path.stat().st_size != expect:
        return False
    raw = np.memmap(str(bin_path), dtype=np.int16, mode="r", shape=(nf, ly, lx))
    try:
        for i in (0, nf // 4, nf // 2, (3 * nf) // 4, nf - 1):
            if int(np.max(np.abs(raw[i]))) != 0:
                return True
    finally:
        del raw
    return False


def convert_one(v: int) -> Path:
    plane = view_plane_dir(v, DATA_ROOT)
    ops = load_npy(plane / "ops.npy", allow_pickle=True).item()
    ly, lx = int(ops["Ly"]), int(ops["Lx"])
    nf = int(ops["nframes"])
    out_bin = plane / "data.bin"

    if _bin_ok(out_bin, nf, ly, lx):
        print("view%d: skip existing %s (%d bytes)" % (v, out_bin, out_bin.stat().st_size))
        return out_bin

    tiff_path = view_tiff_path(v, DATA_ROOT)
    expect = nf * ly * lx * 2

    yoff = ops.get("yoff")
    xoff = ops.get("xoff")
    yoff_a = None if yoff is None else np.asarray(yoff).ravel()
    xoff_a = None if xoff is None else np.asarray(xoff).ravel()

    print("view%d: streaming %s -> %s  (%d x %d x %d)" % (v, tiff_path, out_bin, nf, ly, lx))
    n_nz = 0
    with tifffile.TiffFile(str(tiff_path)) as tif:
        n_pages = len(tif.pages)
        if n_pages < nf:
            raise RuntimeError("TIFF pages %d < nframes %d" % (n_pages, nf))
        tmp = out_bin.with_suffix(".bin.tmp")
        with open(tmp, "wb") as f:
            for i in range(nf):
                frame = np.asarray(tif.pages[i].asarray())
                if frame.shape != (ly, lx):
                    raise ValueError(
                        "TIFF page %d shape %s != ops %dx%d" % (i, frame.shape, ly, lx)
                    )
                if yoff_a is not None and xoff_a is not None and i < yoff_a.size:
                    dy, dx = int(yoff_a[i]), int(xoff_a[i])
                    if dy != 0 or dx != 0:
                        frame = np.roll(frame, shift=(-dy, -dx), axis=(0, 1))
                clipped = _frame_to_int16(frame)
                if np.any(clipped):
                    n_nz += 1
                clipped.tofile(f)
                del frame, clipped
                if (i + 1) % 500 == 0:
                    print("    page %d/%d" % (i + 1, nf))
                    gc.collect()
        got = tmp.stat().st_size
        if got != expect:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("temp data.bin size %d != expect %d" % (got, expect))
        tmp.replace(out_bin)

    if n_nz == 0:
        raise RuntimeError("view %d converted movie is all zeros" % v)
    print(
        "  wrote %s  shape=(%d,%d,%d) dtype=int16  nonzero_frames=%d/%d"
        % (out_bin, nf, ly, lx, n_nz, nf)
    )
    gc.collect()
    return out_bin


def main() -> None:
    print("DATA_ROOT =", DATA_ROOT)
    for v in range(1, N_VIEWS + 1):
        convert_one(v)
    print("Done. All view data.bin ready.")


if __name__ == "__main__":
    main()
