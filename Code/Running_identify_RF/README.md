# Running speed identification — standalone example

Classify running vs non-running from suite2p calcium traces (3 planes, 190 ROIs after cross-plane deduplication) using RandomForest and contiguous time-block 10-fold cross-validation. 

## Contents

```
RunningSpeed_identify_manual_cv_pack/
  RunningSpeed_identify_cv.ipynb   # time-block CV classifier
  requirements.txt
  README.md
  data/
    suite2p/Plane_*_motionfree/suite2p/plane0/
      F.npy, Fneu.npy, isCell.npy, stat.npy
    Region2_file3_20231230_175745.csv     # treadmill speed
    Motion_CA1run_R2f3.mat                # optional motion sanity plot
```

`ops.npy`, `spks.npy`, and `data.bin` are not included (these notebooks do not read them).

## Run

Open the notebooks from this folder and set the Jupyter working directory to this folder.

```bash
pip install -r requirements.txt
jupyter notebook RunningSpeed_identify_manual_cv.ipynb
jupyter notebook PCA.ipynb
```

The circular-shuffle test (`RUN_CIRCULAR_SHUFFLE = True`, 50 shuffles) is slow. For a quick demo, leave it `False` in the first code cell.

## Data

- Experiment: 2023-12-30 GCaMP8m LinearRunner, Region2 file3
- Planes: z = 17 / 21 / 25 (motion-free, method2 cross-plane deduplication)
- Analysis window: 4500 frames (`F.npy` / `Fneu.npy` are already truncated; no linear detrend)
- Running threshold: 0.03 mm/s in the CV notebook; 

