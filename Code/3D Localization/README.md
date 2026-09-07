# 4-view 3D localization example (CA1 R1 f5)

Python example that localizes neurons from four calcium views into 3D, then aligns them to reconstruction-plane suite2p ROIs. All paths are relative to this folder.

This package omits ~12 GB of raw movies (`Views/*/file_*.tif` and `data.bin`). You can inspect results and re-plot figures. You **cannot** re-run localization or bin recovery unless you put the raw movies back.

## Layout

```text
3D Localization CA1/
  README.md                 # this file
  run_full_pipeline.py      # Python entry point
  run_full_pipeline.m       # MATLAB entry point (addpath matlab/ + lib/)
  code/                     # Python pipeline
  matlab/                   # MATLAB pipeline entry points
  matlab/lib/               # MATLAB helpers (io, psf, localize, match, recover)
  PSF/file_{1-4}.tif        # PSF calibration stacks
  Views/{1-4}/suite2p/      # four-view suite2p (no TIFF / data.bin)
  Planes/Plane{14,19,24,29}/suite2p/
  CA1R1f5Output/            # Python example results (unchanged)
  CA1R1f5OutputMatlab/      # MATLAB outputs
```

## What this package can do

| Task | Works with this package? |
|------|--------------------------|
| Inspect method1 / final 3D catalog and figures | Yes (Python example in `CA1R1f5Output/`) |
| Re-plot the 3D scatter | Yes: `python code/plot_m2_3d_distribution.py` or MATLAB `plot_m2_3d_distribution` after a MATLAB run |
| Re-run matching using existing `localization.csv` | Yes (suite2p `F` / `stat` / `ops` are included; MATLAB also needs `matlab/export_s2p_meta.py` sidecars) |
| Method1 localization, `recover_m2_from_bin`, full pipeline | Needs TIFF or `data.bin` restored |

## Environment

Python 3, from this directory:

```text
pip install -r requirements.txt
```

Packages: `numpy`, `scipy`, `tifffile`, `matplotlib`, `pandas`, `openpyxl`.

suite2p is not required; only its `.npy` outputs are used. `code/wait_for_ram.py` queries physical RAM via the Windows API. On other OS, skip the RAM wait and run the step scripts under `code/` directly.

## Inspect results (no raw movies needed)

1. **Method1 (4-view localization)**
   - `CA1R1f5Output/localization.csv` (**212** neurons kept)
   - `CA1R1f5Output/localization_3d.png`
   - `CA1R1f5Output/neuron_dictionary.npy`

2. **Final delivery: 3D positions of reconstruction-plane neurons**
   - `CA1R1f5Output/match/m2_3d_coverage/m2_3d_catalog.csv`
     **191** neurons after dedupe, all with 3D (`source` is `matched_trusted` or `recovered_bin_refined`)
   - `neurons_3d_scatter.png`, `neurons_z_hist.png`

3. **Coverage summary**
   - `CA1R1f5Output/match/m2_3d_coverage/coverage_summary.json`

Re-plot the 3D scatter:

```text
python code/plot_m2_3d_distribution.py
```

## Full run (restore raw movies first)

Place files at the same relative paths `code/dataset_config.py` expects:

| Path | Contents |
|------|----------|
| `Views/{1-4}/file_{v}.tif` | Four-view calcium movies, 256×256×5000 |
| or `Views/{1-4}/suite2p/plane0/data.bin` | suite2p int16 movies converted from TIFF |
| `Planes/Plane{14,19,24,29}/suite2p/plane0/data.bin` | Needed for bin recovery (recon-plane suite2p npy is already included) |
| `Planes/Plane{id}/SinglePlane{id}.tif` | Optional; recon-plane suite2p is already included |

Then:

```text
cd "3D Localization CA1"
python run_full_pipeline.py
```

`run_full_pipeline.py` converts TIFF to `data.bin` first (skips valid existing bins), waits for free RAM on Windows (~35% of total, at least 5 GB, at most 16 GB), then runs the remaining steps. If RAM stays low it exits after bins and does not start localization.

Step by step (from this directory):

```text
python code/prepare_view_bins_from_tiff.py
python code/run_lr_localize.py
python code/match_linear_runner_to_recon_planes.py
python code/promote_unmatched_via_view_suite2p.py
python code/recover_m2_from_bin.py
python code/refine_z_from_footprints.py
python code/dedupe_m2_and_relocalize.py
python code/export_deduped_suite2p.py
python code/plot_m2_3d_distribution.py
```

Scripts write only under `CA1R1f5Output/`. Original suite2p folders are treated as read-only. Match mode defaults to `loose` (`MATCH_MODE` in `code/match_linear_runner_to_recon_planes.py`).

The example Python results in `CA1R1f5Output/` were produced without `promote_unmatched`. The packaged full pipeline still includes that step so this example and the LinearRunner package share the same command list.

## Method

```text
PSF stacks ──► psf_seg + psf_fit ──► method1 4-view localization
4-view suite2p ────────────────────►              │
                                                  ▼
recon-plane suite2p ──► Hungarian match ◄─────────┘
                                                  │
                     weak / unmatched ROIs ──► data.bin parallax z-search
                                                  │
                                        z refine + plane-depth A/B test
                                                  │
                                        cross-plane dedupe ──► m2_3d_catalog.csv
```

### Parallax calibration

Imaging: zoom 5, 256 px. PSF: zoom 15, 256 px, 2 µm/plane.

```text
FACTOR = (PSF_zoom / Image_zoom) × (PSF_pixel / Image_pixel)
       = (15 / 5) × (256 / 256) = 3.0
```

View-center alignment uses `move = centers / MOVE_DIVISOR` with `MOVE_DIVISOR = 3.0` (same number as `FACTOR`; the original CA1 run passed `FACTOR` into `compute_move`).

### Method1: 4-view localization (`code/run_linear_runner_localize.py`)

1. Segment PSF volumes and fit per-view centroid trajectories vs depth (independent x(z) and y(z)).
2. For each suite2p ROI, sweep depth: apply a parallax shift to the footprint and correlate against the other views.
3. Merge the four view dictionaries by 3D distance and trace correlation.
4. Quality filter uses the best k-of-4 views (k≥2): peak correlation, center spread, trace correlation. This dataset keeps **212** neurons.

### Match to recon planes (`code/match_linear_runner_to_recon_planes.py`)

Recon-plane FOV is 255 px; an affine transform from meanImg bridges it to method1's 256 px. Nominal plane depths (relative to geometric center 21.5):

```text
Plane 14 → −15 µm    Plane 19 → −5 µm    Plane 24 → +5 µm    Plane 29 → +15 µm
(2 µm × (plane_id − 21.5); PLANE_Z_FLIPPED = False)
```

One-to-one Hungarian assignment with cost `w_xy·dxy + w_z·|dz| + w_r·(1−r)`, gated by distance / correlation thresholds. In `loose` mode 207/212 method1 neurons match 228 plane ROIs.

### Fill in 3D for method2

- **Trusted matches** (`matched_trusted`): keep method1 `(col, row, z)`.
- **Weak / unmatched**: warp the plane ROI into view coordinates, then sweep z on the four-view `data.bin` (~±20 µm, 2 µm steps), correlating footprint traces with plane `F`.
- **Refine**: local XY+z search on block-averaged movies, plus a global plane-z sign A/B test. This dataset chose **A** (nominal depths, not flipped).

### Cross-plane dedupe (`code/dedupe_m2_and_relocalize.py`)

Build an XY tree of plane ROIs that already have 3D. Pairs with `d_xy ≤ 10 px` and trace `r ≥ 0.6` are union-find merged; the higher-SNR trace is kept. This dataset: **228 → 191**.

Deduped suite2p is written to `CA1R1f5Output/m2_suite2p_after_dedupe/` without modifying original `Planes/`.

## Dataset constants

From `code/dataset_config.py`:

| Item | Value |
|------|-------|
| Imaging zoom / FOV | 5 / 256 px / 160 µm (`PIXEL_UM` = 0.625) |
| PSF | zoom 15, 256×256, 2 µm/plane |
| `FACTOR` | 3.0 |
| `MOVE_DIVISOR` | 3.0 |
| View centers (1-based row, col) | (138,121), (138,123), (143,124), (145,123) |
| Recon planes | 14 / 19 / 24 / 29 |
| `PLANE_Z_FLIPPED` | False |
| `MATCH_MID_PLANE` | 19 |

Completed-run numbers (already in `CA1R1f5Output/`): method1 kept 212; loose match 207/212 vs 228 plane ROIs; coverage 228→191 (48 trusted + 143 recovered); after dedupe 191, all with 3D.

## MATLAB

A MATLAB port of the same pipeline lives in `matlab/` and writes to `CA1R1f5OutputMatlab/` (Python results in `CA1R1f5Output/` are never overwritten). Ported from the Python code: independent `x(z)`/`y(z)` PSF fits, `FACTOR=3`, `MOVE_DIVISOR=3`, then match / recover / refine / dedupe.

Requires **R2019b+** and the Image Processing Toolbox (`bwlabel`, `imresize`). No File Exchange packages besides `matlab/lib/io/readNPY.m`.

```text
cd "3D Localization CA1"
run_full_pipeline
% or:
addpath matlab
% dataset_config adds matlab/lib/{io,psf,localize,match,recover}
run_localize
```

`matlab/` keeps only the step drivers. Helpers live under `matlab/lib/`:

```text
matlab/
  run_full_pipeline.m  dataset_config.m
  prepare_view_bins_from_tiff.m  run_localize.m
  match_to_recon_planes.m  promote_unmatched.m
  recover_m2_from_bin.m  refine_z_from_footprints.m
  dedupe_m2.m  export_deduped_suite2p.m  plot_m2_3d_distribution.m
  lib/io/        % npy/csv/tiff, suite2p, data.bin memmap
  lib/psf/       % PSF segment/fit, parallax, move
  lib/localize/  % depth search, merge, quality filter
  lib/match/     % XY warp, Hungarian IO, plane-z
  lib/recover/   % bin z-sweep, refine, footprints
```

`run_full_pipeline.m` at the package root adds `matlab/` and `matlab/lib/*` to the path and runs:

```text
prepare_view_bins_from_tiff
run_localize
match_to_recon_planes
promote_unmatched
recover_m2_from_bin
refine_z_from_footprints
dedupe_m2
export_deduped_suite2p
plot_m2_3d_distribution
```

Constants match `code/dataset_config.py` (`matlab/dataset_config.m`). Match mode defaults to `loose`. Catalog / CSV / PNG column names match the Python outputs so the two runs can be compared.

suite2p `stat.npy` / `ops.npy` are pickled; MATLAB cannot load them. A one-shot helper writes numeric sidecars next to each `plane0/`:

```text
python matlab/export_s2p_meta.py
```

That creates `s2p_matlab/` (`xpix.npy`, `ypix.npy`, `lam.npy`, `meanImg.npy`, `ops.json`, …). `load_suite2p.m` reads those plus numeric `F.npy` / `Fneu.npy` / `iscell.npy`. Original npy folders remain the Python source of truth. Sidecars for this package are already generated.

Promote writes view-suite2p copies under `CA1R1f5OutputMatlab/views_suite2p_working/` and does **not** rewrite original `Views/`.

Without TIFF/`data.bin`, MATLAB can still re-run match if `localization.csv` exists from a completed MATLAB run, and can plot from its own `m2_3d_catalog.csv`. Localization and bin recovery need the raw movies, same as Python. There is no RAM-wait helper; MATLAB proceeds immediately.

There is no MATLAB-vs-`Neuron_Dictionary.mat` comparison in this package.

## What was left out

Package size is about **0.55 GB** vs ~**13 GB** with raw movies. Not included:

- `Views/{1-4}/file_{v}.tif` and `suite2p/plane0/data.bin`
- `Planes/Plane{14,19,24,29}/SinglePlane*.tif` and plane `data.bin`
- suite2p `Fall.mat`
- bulky output folders: `neuron_traces/`, `recovered_from_bin/pair_plots/`
- `Neuron_Dictionary.mat`, `vs_NeuronDictionary/`, `archive_autoCenter/`, compare scripts

To re-run the full stack, restore TIFF or `data.bin` at the paths above. No code changes are required.
