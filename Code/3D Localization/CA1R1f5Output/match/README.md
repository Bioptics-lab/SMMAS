# Match pipeline outputs (`CA1R1f5Output/match`)

## Plane depth convention

A/B test (after full refine) selected **A**: plane_z={14: -15.0, 19: -5.0, 24: 5.0, 29: 15.0}. See `m2_3d_coverage/z_refined/plane_z_ab_test.json`.


## Result summary (this run)

| Stage | Result |
|-------|--------|
| Method1 localize | 211 kept |
| Loose match | 204 / m1=211 / m2=228 |
| Recover | 227/228 with 3D (169 recovered + 58 trusted) |
| After refine | convention A |
| After dedupe | **191/191** with 3D (47 trusted + 144 refined) |

Key paths:

- `loose/matches.csv`
- `m2_3d_coverage/m2_3d_catalog.csv`
- `m2_3d_coverage/dedupe/`
- `../m2_suite2p_after_dedupe/`
- `m2_3d_coverage/neurons_3d_scatter.png`
