Deduped method2 suite2p exports (cross-plane duplicates removed; higher SNR kept).
Original suite2p folders were NOT modified.

Layout:
  Plane14/suite2p/plane0/
  Plane19/suite2p/plane0/
  Plane24/suite2p/plane0/
  Plane29/suite2p/plane0/

Each plane0 contains: F.npy, Fneu.npy, spks.npy, stat.npy, iscell.npy, ops.npy
(no data.bin — use the original plane's data.bin / meanImg in ops if needed).

ROI indices are reindexed 0..N-1 per plane; see roi_index_map.csv for old_roi_idx.
Source kept list: F:\SMMAS-Revision\3DLocalizationCA1R1f5\CA1R1f5Output\match\m2_3d_coverage\dedupe\kept_rois.csv
Total kept ROIs: 191
