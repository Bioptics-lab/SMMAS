# -*- coding: utf-8 -*-
"""Plot 3D neuron scatter + z-axis count histogram from m2_3d_catalog.

XY in µm: zoom5 imaging FOV is 160×160 µm on 256×256 px.
"""
from __future__ import annotations

import csv

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from dataset_config import FOV_UM, OUT_DIR as PROJECT_OUT, PIXEL_UM, PLANE_Z_BASE

CATALOG = PROJECT_OUT / "match" / "m2_3d_coverage" / "m2_3d_catalog.csv"
OUT_DIR = CATALOG.parent

# --- scatter style (same visual language as LinearRunner NeuronPos plots) ---
# Rotate -90° vs NeuronPos azim so the XY origin (0, 0) sits at the bottom-front.
VIEW_AZIM = 327.62631579 - 360.0 - 90.0  # ≈ -122.37
VIEW_ELEV = 19.0
FACE_BLUE = (0.54117647, 0.64313725, 0.90196078)
EDGE_BLUE = (0.14901961, 0.14901961, 0.14901961)
FACE_RED = (1.0, 0.2, 0.2)
SIZE_BLUE = 60
XLIM = (0.0, FOV_UM)
YLIM = (0.0, FOV_UM)
ZLIM = (-25.0, 25.0)
Z_PLOT_MAX_UM = 30.0  # omit ROIs with |z| > this from scatter plots
XTICKS = [0.0, round(FOV_UM / 2.0), round(FOV_UM)]
YTICKS = [0.0, round(FOV_UM / 2.0), round(FOV_UM)]
ZTICKS = [-25, 0, 25]
SHOW_LEGEND = False
GRID_COLOR = (0.72, 0.72, 0.72, 0.55)
GRID_LW = 0.4
PANE_EDGE = (0.68, 0.68, 0.68, 0.7)
# Scatter plots -z; frames sit at -PLANE_Z_BASE so they line up with displayed z.
_PLANE_COLORS = ("hotpink", "darkcyan", "skyblue", "darkseagreen")
PLANE_FRAMES = tuple(
    (-z, _PLANE_COLORS[i % len(_PLANE_COLORS)])
    for i, (_pid, z) in enumerate(sorted(PLANE_Z_BASE.items()))
)
SCATTER_FONT_SCALE = 1.5
PLANE_FRAME_LW = 1.6


def _set_arial(scale: float = 1.0) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9.0 * scale,
            "axes.labelsize": 9.9 * scale,
            "axes.linewidth": 0.5,
            "xtick.labelsize": 9.0 * scale,
            "ytick.labelsize": 9.0 * scale,
            "lines.linewidth": 0.5,
        }
    )


def load_catalog(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = []
    for r in rows:
        if r.get("source") == "failed":
            continue
        if r.get("z_um") in ("", None) or r.get("col") in ("", None):
            continue
        out.append(
            {
                "col": float(r["col"]),
                "row": float(r["row"]),
                "z_um": float(r["z_um"]),
                "source": str(r["source"]),
                "plane": int(r["m2_plane"]),
                "roi": int(r["m2_roi"]),
            }
        )
    return out


def load_vip_old_keys() -> set[tuple[int, int]]:
    """No VIP overlay on this CA1 package (LinearRunner NeuronPos list does not apply)."""
    return set()


def _scatter3d(ax, x, y, z, vip=None, *, plane_frames: bool = False) -> None:
    if vip is None:
        vip = np.zeros(len(x), dtype=bool)
    else:
        vip = np.asarray(vip, dtype=bool)
    rest = ~vip
    common = dict(
        s=SIZE_BLUE,
        edgecolors=[EDGE_BLUE],
        linewidths=0.5,
        depthshade=False,
        marker="o",
    )
    if rest.any():
        ax.scatter(x[rest], y[rest], z[rest], c=[FACE_BLUE], zorder=2, **common)
    if vip.any():
        ax.scatter(x[vip], y[vip], z[vip], c=[FACE_RED], zorder=3, **common)
    if plane_frames:
        _draw_plane_frames(ax)
    _style_axes_3d(ax)


def _style_axes_3d(ax) -> None:
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_zlim(*ZLIM)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)
    ax.set_zticks(ZTICKS)
    scale = float(mpl.rcParams["font.size"]) / 9.0
    pad = 8.0 * scale
    fs = 9.0 * scale
    ls = 9.9 * scale
    ax.set_xlabel("Lateral position (\u00b5m)", labelpad=pad, fontsize=ls)
    ax.set_ylabel("Lateral position (\u00b5m)", labelpad=pad, fontsize=ls)
    ax.set_zlabel("Axial position (\u00b5m)", labelpad=pad, fontsize=ls)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        width=0.5,
        length=3,
        labelsize=fs,
    )
    ax.tick_params(axis="z", which="major", labelsize=fs)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor(PANE_EDGE)
        axis.pane.set_linewidth(0.45)
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = GRID_LW
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.2
        axis.set_tick_params(labelsize=fs, pad=3.0 * scale)
    ax.grid(True)
    ax.set_box_aspect((1.0, 1.0, 0.57))


def _draw_plane_frames(ax) -> None:
    x0, x1 = XLIM
    y0, y1 = YLIM
    for z0, color in PLANE_FRAMES:
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            [z0] * 5,
            color=color,
            lw=PLANE_FRAME_LW,
            zorder=4,
        )


def main() -> None:
    _set_arial()
    rows = load_catalog(CATALOG)
    if not rows:
        raise SystemExit("no neurons with 3D coords in %s" % CATALOG)

    x = np.array([r["col"] for r in rows]) * PIXEL_UM
    y = np.array([r["row"] for r in rows]) * PIXEL_UM
    z = np.array([r["z_um"] for r in rows])
    src = [r["source"] for r in rows]
    vip_keys = load_vip_old_keys()
    is_vip = np.array([(r["plane"], r["roi"]) in vip_keys for r in rows], dtype=bool)
    in_z = np.abs(z) <= Z_PLOT_MAX_UM
    n_omit = int((~in_z).sum())
    x_s, y_s, z_s = x[in_z], y[in_z], z[in_z]
    vip_s = is_vip[in_z]
    print(
        "VIP highlighted %d / %d (omitted |z|>%.0f: %d)"
        % (int(vip_s.sum()), int(is_vip.sum()), Z_PLOT_MAX_UM, int((is_vip & ~in_z).sum()))
    )

    # --- 3D scatter: z flipped; plane frames from -PLANE_Z_BASE ---
    _set_arial(SCATTER_FONT_SCALE)
    fig = plt.figure(figsize=(8.6, 5.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _scatter3d(ax, x_s, y_s, -z_s, vip=vip_s, plane_frames=True)
    fig.tight_layout()
    p3d = OUT_DIR / "neurons_3d_scatter.png"
    fig.savefig(str(p3d), dpi=160, facecolor="white")
    plt.close(fig)
    print("wrote", p3d, "n=%d (omitted |z|>%.0f: %d)" % (int(in_z.sum()), Z_PLOT_MAX_UM, n_omit))
    _set_arial(1.0)

    # --- z histogram (unchanged role; restyle lightly) ---
    fig, ax = plt.subplots(figsize=(7.0, 4.2), facecolor="white")
    z_min = float(np.floor(z.min() / 2) * 2)
    z_max = float(np.ceil(z.max() / 2) * 2)
    bins = np.arange(z_min, z_max + 2.01, 2.0)
    sources = [
        s
        for s in ("matched_trusted", "recovered_bin_refined", "recovered_bin")
        if any(x == s for x in src)
    ]
    data = [z[np.array([x == s for x in src])] for s in sources]
    colors = [FACE_RED if s == "matched_trusted" else FACE_BLUE for s in sources]
    labels = ["%s (n=%d)" % (s, len(d)) for s, d in zip(sources, data)]
    ax.hist(data, bins=bins, stacked=True, color=colors, label=labels, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Axial position (\u00b5m)")
    ax.set_ylabel("neuron count")
    ax.set_title("Neuron count along z (bin=2 \u00b5m, n=%d)" % len(rows))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    ph = OUT_DIR / "neurons_z_hist.png"
    fig.savefig(str(ph), dpi=140, facecolor="white")
    plt.close(fig)
    print("wrote", ph)

    print(
        "z range [%.1f, %.1f] um  median=%.1f  xy um range x[%.1f,%.1f] y[%.1f,%.1f]"
        % (
            float(z.min()),
            float(z.max()),
            float(np.median(z)),
            float(x.min()),
            float(x.max()),
            float(y.min()),
            float(y.max()),
        )
    )


if __name__ == "__main__":
    main()
