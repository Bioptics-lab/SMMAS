# -*- coding: utf-8 -*-
"""Dataset constants (zoom5 imaging, 256 px, zoom15 PSF).

Scripts live in code/; data, PSF, and outputs sit in the package root.
"""
from __future__ import annotations

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT = CODE_DIR.parent
DATA_ROOT = PROJECT / "Views"
PSF_DIR = PROJECT / "PSF"
PLANES_DIR = PROJECT / "Planes"
OUT_DIR = PROJECT / "CA1R1f5Output"

# Parallax / estimate_depth: (PSF_zoom / Image_zoom) * (PSF_pixel / Image_pixel)
# = (15/5) * (256/256) = 3.0
FACTOR = 3.0
# CA1 original run passed FACTOR into compute_move; keep MOVE_DIVISOR equal.
MOVE_DIVISOR = 3.0
PSF_DZ_UM = 2.0
IMAGE_ZOOM = 5.0
PSF_ZOOM = 15.0
FOV_PX = 256.0
FOV_UM = 160.0
PIXEL_UM = FOV_UM / FOV_PX  # 0.625 µm/px

# MATLAB CA1_R1_f5: XYcenter = [138 121; 138 123; 143 124; 145 123]  (1-based row, col)
VIEW_CENTERS_OVERRIDE: list[tuple[int, int]] = [
    (138, 121),
    (138, 123),
    (143, 124),
    (145, 123),
]
# zcenter=33, Reconstruct_range=40 → 13:53 inclusive
PSF_ZRANGE_1BASED = range(13, 54)
CROP_HALF_Y = 59  # 2*59+1 = 119
CROP_HALF_X = 60  # 2*60+1 = 121

# Recon planes 14/19/24/29; z = 2 µm * (id - 21.5)
PLANE_Z_BASE: dict[int, float] = {
    14: -15.0,
    19: -5.0,
    24: 5.0,
    29: 15.0,
}
PLANE_Z_FLIPPED = False
MATCH_MID_PLANE = 19

T_STRIDE = 10
TRACE_T_STRIDE = 10

FORCE_RECALIB = False
REFILTER_ONLY = False


def rel_to_project(path: Path | str) -> str:
    """Return a portable relative path from the package root (forward slashes)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def view_dir(v: int, data_root: Path | None = None) -> Path:
    """Views/{v} (this dataset) or Views/file_{v} (Ai162-style fallback)."""
    root = DATA_ROOT if data_root is None else Path(data_root)
    candidates = (
        root / str(v),
        root / ("file_%d" % v),
    )
    for c in candidates:
        if (c / "suite2p" / "plane0").is_dir():
            return c
    raise FileNotFoundError(
        "no suite2p plane0 for view %d under %s (tried %d and file_%d)"
        % (v, root, v, v)
    )


def view_plane_dir(v: int, data_root: Path | None = None) -> Path:
    return view_dir(v, data_root) / "suite2p" / "plane0"


def view_tiff_path(v: int, data_root: Path | None = None) -> Path:
    d = view_dir(v, data_root)
    p = d / ("file_%d.tif" % v)
    if p.is_file():
        return p
    raise FileNotFoundError(p)


def plane_suite2p_dir(plane_id: int) -> Path:
    return PLANES_DIR / ("Plane%d" % plane_id) / "suite2p" / "plane0"


def move_divisor_from_coef(coef) -> float:
    files = getattr(coef, "files", None)
    if files is not None and "move_divisor" in files:
        return float(coef["move_divisor"])
    return float(MOVE_DIVISOR)
