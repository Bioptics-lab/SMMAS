# -*- coding: utf-8 -*-
"""Export neurons_3d_scatter.png source data (plotted points only) to Excel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from dataset_config import FOV_UM, PIXEL_UM
from plot_m2_3d_distribution import (
    CATALOG,
    OUT_DIR,
    Z_PLOT_MAX_UM,
    load_catalog,
    load_vip_old_keys,
)

OUT_XLSX = OUT_DIR / "neurons_3d_scatter_source_data.xlsx"


def plotted_rows() -> list[dict]:
    rows = load_catalog(CATALOG)
    if not rows:
        raise SystemExit("no neurons with 3D coords in %s" % CATALOG)
    vip_keys = load_vip_old_keys()
    out = []
    for r in rows:
        z_um = float(r["z_um"])
        if abs(z_um) > Z_PLOT_MAX_UM:
            continue
        highlighted = (r["plane"], r["roi"]) in vip_keys
        out.append(
            {
                "point_id": len(out) + 1,
                "lateral_x_um": float(r["col"]) * PIXEL_UM,
                "lateral_y_um": float(r["row"]) * PIXEL_UM,
                "axial_z_um": -z_um,
                "marker_color": "red" if highlighted else "blue",
                "highlighted": "yes" if highlighted else "no",
                "imaging_plane": int(r["plane"]),
                "roi_index": int(r["roi"]),
                "localization_source": str(r["source"]),
            }
        )
    return out


def _autosize(ws) -> None:
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        width = 12
        for cell in col:
            if cell.value is None:
                continue
            width = max(width, min(len(str(cell.value)) + 2, 36))
        ws.column_dimensions[letter].width = width


def write_xlsx(records: list[dict], path: Path) -> None:
    df = pd.DataFrame(records)
    notes = pd.DataFrame(
        [
            ["figure", "neurons_3d_scatter.png"],
            ["n_points", str(len(df))],
            ["n_highlighted_red", str(int((df["highlighted"] == "yes").sum()))],
            ["units", "micrometers (um)"],
            ["lateral_x_um", "col (px) * pixel_size; pixel_size = FOV_UM / 512"],
            ["lateral_y_um", "row (px) * pixel_size"],
            ["axial_z_um", "as plotted: -catalog z_um (z axis is flipped in the figure)"],
            ["FOV_UM", str(FOV_UM)],
            ["pixel_size_um", str(PIXEL_UM)],
            ["include_rule", "catalog neurons with valid xyz; omit |catalog z| > %.1f um" % Z_PLOT_MAX_UM],
            ["axis_limits", "x,y: 0-%.0f um; z: -20 to 20 um" % FOV_UM],
            ["marker_color", "red = highlighted example neurons; blue = all other plotted neurons"],
            ["plane_frames", "reference frames at +8 / 0 / -8 um are guides, not data points"],
            ["catalog", str(CATALOG)],
        ],
        columns=["item", "value"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="SourceData", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            header_font = Font(bold=True)
            for cell in sheet[1]:
                cell.font = header_font
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            _autosize(sheet)
    print("wrote", path, "n=%d" % len(df))


def main() -> None:
    records = plotted_rows()
    write_xlsx(records, OUT_XLSX)


if __name__ == "__main__":
    main()
