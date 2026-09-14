# Forward distance-window evaluation

Distance episodes are fresh localization trials over cached front-end outputs,
not independent recordings or freshly rerun trackers. Each episode uses a
uniform prior over the **original parent region**, the unchanged catalog,
and newly generated calibrated planar IMU error. No reverse traversal,
truth-centered initialization, or final/fixed-lag smoothing is used.

For `N` windows over a parent of traveled distance `D`, the start stride is
`D / (N + 1)` and the window length is twice that stride (50% overlap).
Endpoints snap forward to recorded keyframes and are inclusive. A single
window retains the full parent and its original odometry seed stream.
Distance is accumulated along the reference path, not endpoint displacement;
there is no minimum-duration cutoff.

## First roster (35 episodes)

| Parent | Windows |
| --- | ---: |
| Flevoland | 3 |
| Boston Harbor leg 1 | 3 |
| Boston Harbor leg 2 | 1 |
| Boston Harbor leg 3 | 3 |
| Franconia | 3 |
| Portland legs 1, 2, 3 | 3 each |
| Charles River | 7 |
| Pohang | 3 |
| Mount Washington legs 1, 2, 3 | 1 each |

The initial pass compares the current (non-deduplicated) tracker against
no tracking, with range gates enabled and odometry seed 0: 70 evaluations.
Pair episode keyframe bounds, motion sources, seeds, and all scientific
filter settings across methods. The seed stream includes the parent dataset
and episode keyframe bounds, never the tracker/artifact version.

## Boundary evidence and portability

Keep a track only when its **source birth** and natural release both fall
inside the window. Checking the first accepted bearing is insufficient:
the semantic audit may have used earlier observations. Do not clip bearings
and reuse an out-of-window audit/match, or flush a crossing track early at a
synthetic endpoint. Parent EOF releases are retained only in windows that
reach that EOF. The plan records retained track IDs for each result.

`build_distance_episodes` verifies the bound source tracking artifacts once
and writes a portable plan bound to the immutable localization input and
the exact existing release sidecar. Workers need the localization input,
release sidecar, matching tables, and plan; they do not need to copy or rerun
tracking, auditing, or matching. Treat the generated plan as a versioned
evaluation input and preserve its bytes/hash with the run manifest.

```bash
bazel run //experimental/overhead_matching/swag/farfield/localization:build_distance_episodes -- \
  --input_dir /data/farfield_matching/artifacts/localization_inputs/DATASET/VERSION \
  --release_schedule /path/to/parent_release_schedule.json \
  --count 3 --out /path/to/distance_episodes.json

bazel run //experimental/overhead_matching/swag/farfield/localization:grid_filter -- \
  --input_dir /data/farfield_matching/artifacts/localization_inputs/DATASET/VERSION \
  --release_schedule /path/to/parent_release_schedule.json \
  --tables_override /path/to/parent_matching_tables.divided.json \
  --episode_plan /path/to/distance_episodes.json --episode_index 0 \
  --odometry_profile epson_mg570_calibrated_planar_v1 --odometry_seed 0 \
  --track_joint 1 --smoother none --smooth_lag 0 --smooth_lags '' \
  --out /path/to/episode.causal.json
```

The example shows episode-specific arguments only; retain the full resolved
baseline scientific configuration when launching experiments. Odometry is
generated **after slicing clean motion**, with fresh gyro/accelerometer bias,
velocity error, and position error. It is not a slice of a drifting parent
odometry realization. Ground-truth coordinates and map coordinates are not
translated or used to narrow the prior.

Report causal distance-normalized `dn_mass_500` and `dn_mass_100`, with per-
episode scores and acquisition diagnostics. Average episodes within their
parent first; overlapping windows and seeds are not independent datasets.
Keep recording-level grouping when estimating uncertainty.
