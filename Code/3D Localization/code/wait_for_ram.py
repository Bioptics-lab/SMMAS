# -*- coding: utf-8 -*-
"""Poll available physical RAM until it is high enough for localization."""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def mem_status() -> tuple[float, float]:
    """Return (total_gb, avail_gb)."""
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise OSError("GlobalMemoryStatusEx failed")
    scale = 1024.0 ** 3
    return float(stat.ullTotalPhys) / scale, float(stat.ullAvailPhys) / scale


def avail_gb() -> float:
    return mem_status()[1]


def default_min_avail_gb() -> float:
    """16 GB free is impossible on a 16 GB machine. Cap at 35% of total, floor 5 GB."""
    total, _ = mem_status()
    return min(16.0, max(5.0, 0.35 * total))


def wait_for_ram(
    min_avail_gb: float | None = None,
    poll_s: float = 30.0,
    stable_hits: int = 2,
    timeout_s: float = 3.0 * 3600.0,
) -> bool:
    """Return True if available RAM stayed >= min_avail_gb for `stable_hits` polls.

    Return False on timeout. Cannot see another Cursor window; this only watches
    system available physical memory.
    """
    total, _ = mem_status()
    if min_avail_gb is None:
        min_avail_gb = default_min_avail_gb()
    t0 = time.time()
    hits = 0
    print(
        "wait_for_ram: total=%.1f GB  need %.1f GB free, poll=%.0fs, timeout=%.0f min"
        % (total, min_avail_gb, poll_s, timeout_s / 60.0),
        flush=True,
    )
    while True:
        gb = avail_gb()
        elapsed = time.time() - t0
        print("  avail=%.2f GB  elapsed=%.0fs  hits=%d/%d" % (gb, elapsed, hits, stable_hits), flush=True)
        if gb >= min_avail_gb:
            hits += 1
            if hits >= stable_hits:
                print("wait_for_ram: ready (%.2f GB free)" % gb, flush=True)
                return True
        else:
            hits = 0
        if elapsed >= timeout_s:
            print(
                "wait_for_ram: TIMEOUT after %.0fs (last avail=%.2f GB, need %.1f)"
                % (elapsed, gb, min_avail_gb),
                flush=True,
            )
            return False
        time.sleep(poll_s)


if __name__ == "__main__":
    ok = wait_for_ram()
    raise SystemExit(0 if ok else 2)
