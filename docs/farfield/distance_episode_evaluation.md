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

## Revised roster (65 subtracks plus 13 full trajectories)

| Parent | Subtracks | Full trajectories |
| --- | ---: | ---: |
| Flevoland | 5 | 1 |
| Boston Harbor legs 1, 2, 3 | 5 each | 1 each |
| Franconia | 5 | 1 |
| Portland legs 1, 2, 3 | 5 each | 1 each |
| Charles River | 5 | 1 |
| Pohang | 5 | 1 |
| Mount Washington legs 1, 2, 3 | 5 each | 1 each |

The five windows have target length `D / 3`, with starts at
`0, D / 6, D / 3, D / 2, 2D / 3`. Lengths and spacing are approximately
equal after keyframe snapping. Apply the same rule to the four hiking
recordings, retaining sparse windows as part of the evaluation. Generate a
separate `--count 1` plan for each full trajectory; it is not a sixth member
of the five-window plan.

The revised pass compares the detection-first/audited-track hybrid in
[#726](https://github.com/ewfuentes/robot/pull/726) against no tracking, with
range gates enabled and odometry seed 0: **156 evaluations** (130 subtrack
results and 26 full-trajectory results). #726 supplies the hybrid implementation;
this PR supplies the distance windows and containment policy.
Pair episode keyframe bounds, motion sources, seeds, and all scientific
filter settings across methods. The seed stream includes the parent dataset
and episode keyframe bounds, never the tracker/artifact version.
Seed 0 does not repeat the same noise across windows: the dataset and bounds
select distinct random streams, including freshly drawn gyro/accelerometer
biases and white noise. Both methods share the same realization for a given
window. Full runs retain the parent dataset stream. Window spread therefore
includes both starting-position effects and IMU realization effects; it is
not a pure starting-position variance estimate.
This replaces the original 35-episode protocol; existing plans and results
under `260914_distance_episodes` describe that original run and must not be
overwritten or relabeled as the revised evaluation.

## Boundary evidence and portability

Keep a track only when its **source birth** and natural release both fall
inside the window. Checking the first accepted bearing is insufficient:
the semantic audit may have used earlier observations. Do not clip bearings
and reuse an out-of-window audit/match, or flush a crossing track early at a
synthetic endpoint. Parent EOF releases are retained only in windows that
reach that EOF. The plan records retained track IDs for each result.

With #726, this discards the **audited joint factor**, not the underlying raw
in-window detections. Raw detections enter at their own keyframes; only retained
audits replace their source-support factors at natural release. A crossing
track's detections remain unless another retained audit claims them. Previously
emitted causal scores are unchanged by replay. This addresses the raw-evidence
loss of the older delayed-tracking comparison, without assuming that the
remaining difference in audited evidence has negligible performance impact.

Clipping bearings while retaining a whole-track audit is not equivalent to
fresh causal perception: the audit dossier includes the founding observation,
whole-track tag/name histories, and lifetime/closure information, beyond its
sampled image chips. Such clipping can instead be a separately labeled
evaluation of localization over cached full-recording perception. It must not
be reported as the boundary-contained causal protocol above.

### Boundary retention census

Computed from the original run's pinned inputs, source births, and release
schedules using the revised counts and `distance_episodes.windows`. These
are tracklets eligible for localization, not unique physical landmarks or all
raw tracker outputs. Each cell is **kept / discarded for crossing a boundary**,
in start-position order. A crossing track satisfies `birth <= end` and
`release >= start` but is not fully contained. Tracks wholly outside a window
are excluded from these counts; their count is `parent - kept - crossing`.

| Parent | Parent tracks | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flevoland | 160 | 44 / 1 | 45 / 20 | 54 / 9 | 44 / 21 | 53 / 8 |
| Boston Harbor 1 | 93 | 44 / 4 | 27 / 13 | 21 / 7 | 15 / 11 | 22 / 5 |
| Boston Harbor 2 | 62 | 11 / 6 | 5 / 13 | 7 / 9 | 14 / 12 | 35 / 5 |
| Boston Harbor 3 | 166 | 51 / 5 | 22 / 11 | 20 / 10 | 33 / 18 | 86 / 6 |
| Franconia | 132 | 50 / 1 | 41 / 9 | 45 / 3 | 41 / 4 | 34 / 3 |
| Portland 1 | 196 | 65 / 14 | 49 / 17 | 44 / 19 | 45 / 20 | 68 / 5 |
| Portland 2 | 248 | 92 / 9 | 49 / 30 | 62 / 29 | 80 / 22 | 67 / 18 |
| Portland 3 | 213 | 71 / 8 | 42 / 19 | 38 / 14 | 40 / 16 | 90 / 6 |
| Charles River | 132 | 24 / 8 | 44 / 9 | 56 / 11 | 39 / 12 | 41 / 4 |
| Pohang | 305 | 136 / 5 | 85 / 10 | 61 / 15 | 67 / 21 | 93 / 10 |
| Washington 1 | 25 | 2 / 1 | 2 / 2 | 1 / 3 | 9 / 5 | 19 / 2 |
| Washington 2 | 86 | 27 / 4 | 25 / 9 | 31 / 7 | 25 / 8 | 21 / 3 |
| Washington 3 | 129 | 29 / 2 | 32 / 4 | 47 / 10 | 46 / 8 | 45 / 6 |

Across the 65 subdivided windows: **2,773 kept, 639 boundary-discarded**
(18.7% of intersecting track/window instances). Each full run keeps every
track in the parent column and discards none. Overlapping windows count a parent
track repeatedly; these totals are not counts of distinct tracks. Intersection
of a track lifetime does not guarantee an accepted bearing in that window.

The no-tracking arm represents individual detections as zero-duration tracks,
so it loses none at boundaries. For completeness, its kept counts are:

| Parent | Parent detections | Window 1 | Window 2 | Window 3 | Window 4 | Window 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flevoland | 3,081 | 933 | 1,081 | 1,098 | 1,287 | 1,060 |
| Boston Harbor 1 | 1,388 | 657 | 386 | 376 | 360 | 359 |
| Boston Harbor 2 | 781 | 188 | 142 | 140 | 230 | 460 |
| Boston Harbor 3 | 2,521 | 667 | 471 | 486 | 746 | 1,373 |
| Franconia | 1,472 | 583 | 500 | 515 | 450 | 380 |
| Portland 1 | 2,445 | 980 | 778 | 646 | 690 | 831 |
| Portland 2 | 2,867 | 1,141 | 802 | 817 | 1,023 | 922 |
| Portland 3 | 2,532 | 837 | 696 | 726 | 656 | 974 |
| Charles River | 2,257 | 639 | 916 | 891 | 691 | 737 |
| Pohang | 6,976 | 1,930 | 2,007 | 1,953 | 2,461 | 3,101 |
| Washington 1 | 222 | 36 | 50 | 56 | 102 | 133 |
| Washington 2 | 980 | 317 | 352 | 353 | 311 | 314 |
| Washington 3 | 1,102 | 242 | 277 | 405 | 488 | 467 |

The hybrid initially uses these same raw detections. Across all subtracks,
7,066 raw detection/window instances belong to crossing audited tracks;
6,736 are unclaimed by retained audits and remain as raw factors. The other
330 are shared supports replaced by another retained audit, not dropped
because of the crossing track. Counts come from #726's source-member maps.

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
  --count 5 --out /path/to/distance_episodes.json

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
episode scores and acquisition diagnostics. For each parent and method, report
the **full-trajectory causal score separately** from the five-subtrack mean
and standard deviation. Include every individual subtrack score. Never mix
the full score into the subtrack mean or substitute a final/fixed-lag smoothed
score. Keep full-trajectory and subtrack aggregate tables separate; average
subtracks within their parent first. Overlapping windows and seeds are not
independent datasets. Keep recording-level grouping when estimating uncertainty.
