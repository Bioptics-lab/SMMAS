# PWA Example: Multi-View Light-Field Aberration Correction

Example MATLAB pipeline for **piecewise affine (PWA) correction** of a 4-view Fourier light-field / SMMAS acquisition. Fluorescent beads are used to measure field-dependent displacements across views, reconstruct a low-order Zernike wavefront, warp the four views into alignment, and optionally reconstruct a 3D volume with Richardson–Lucy deconvolution.

Experimental image stacks are **not** bundled (they can be large). Put your own files into `data/` using the filenames below.

## Folder map

```text
PWA_example/
  DisplacementExtraction.m
  WavefrontFitting.m
  PWACorrection.m
  PWACorrectionConfirm.m
  ProcessPipeline_patch_every1frames_SingleLayerBeads.m
  ProcessPipeline_patch_every1frames_SingleLayerBeadsPWA.m
  decrosstalk_timeSequence.m
  createFitx.m
  createFit_f.m
  psf_seg_TPLFM.m
  psf_seg_TPLFM_fit.m
  DisplacementExtractionLocal.m  % local ±20 matching, no initial view offset
  data/
    views/                  % reconstructed reference + four raw views
    PSF/PSF_stack/          % four-view PSF stacks
    ReconstructOrigin/      % uncorrected views for reconstruction
    ReconstructPWA/         % PWA-corrected views for reconstruction
    crosstalk/              % mixing matrix for decrosstalk
```

## Suggested run order

1. **Preprocess** (optional, depending on your raw data)
   - `decrosstalk_timeSequence.m` — unmix four interleaved channels with matrix `S`.
2. **Measure displacements**
   - Load `view0Image` (reconstructed reference) and `view1Image`–`view4Image` into the MATLAB workspace, then run `DisplacementExtraction.m`.
   - File-selection dialogs are commented out, matching the original scripts. Uncomment them if you prefer `uigetfile`.
3. **Wavefront and PWA correction**
   - `WavefrontFitting.m` — Zernike wavefront from bead displacements (center FOV, then an 11×11 spatial grid).
   - `PWACorrection.m` — warp each view with the fitted displacement field; writes `PWA1.tif`–`PWA4.tif` to the current working directory. Copy those files into `data/ReconstructPWA/` before reconstruction.
   - `PWACorrectionConfirm.m` — compare rigid translation vs. PWA warp + translation.
4. **3D reconstruction**
   - `ProcessPipeline_patch_every1frames_SingleLayerBeads.m` — RL deconvolution of uncorrected views (`data/ReconstructOrigin/`).
   - `ProcessPipeline_patch_every1frames_SingleLayerBeadsPWA.m` — same pipeline on PWA-corrected views (`data/ReconstructPWA/`).

## Expected data filenames

Place files under `data/` as follows.

| Folder | Files |
|---|---|
| `data/views/` | `view0.tif` (reference), `view1.tif`–`view4.tif`; optional `F_00002_4.tif` for stripe removal |
| `data/PSF/PSF_stack/` | `file_1.tif`–`file_4.tif` (PSF stacks). The pipeline writes `PSF_1205.mat` here. Optional `PSF1_00004.tif` for decrosstalk. |
| `data/ReconstructOrigin/` | `processed_File1.tif`–`processed_File4.tif` |
| `data/ReconstructPWA/` | `PWA1.tif`–`PWA4.tif` |
| `data/crosstalk/` | `mtxS_Beads_1127.mat` containing variable `S` |

Scripts that use `fileparts(mfilename('fullpath'))` resolve these paths relative to this example folder. Run them as scripts (F5 / `Run`), not by pasting into the Command Window.

PWA scripts (`DisplacementExtraction`, `WavefrontFitting`, `PWACorrection`, `PWACorrectionConfirm`) still expect workspace variables:

- `view0Image` — reconstructed reference
- `view1Image`, `view2Image`, `view3Image`, `view4Image` — the four views

## Dependencies

- MATLAB Image Processing Toolbox (`imresize`, `bwlabel`, `regionprops`, `imtranslate`, `Tiff`, …)
- MATLAB Curve Fitting Toolbox (`fit`, `fittype`, `prepareSurfaceData`)
- External helpers **not included here**:
  - `loadtiff` / `saveastiff` (TIF I/O)
  - `FLFM_deconRL_GPU` (GPU Richardson–Lucy reconstruction; required by the two `ProcessPipeline_*.m` scripts)

Without `FLFM_deconRL_GPU`, displacement extraction, wavefront fitting, and PWA warping still run; only the 3D reconstruction step will fail.
