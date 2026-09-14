# 2026-09-12 localization experiments (harel/loc-experiments)

Results directory: `/data/farfield_matching/runs/260912_harel_experiments/`
(`index.html` links a grid_viewer page per run; `PLAN.md` is the improvement
plan; figures listed below). All numbers: seed 0, natural release, uniform
prior over the catalog region, no heading, no truth in the filter,
`dn_mass_500` (posterior mass within 500 m, distance-normalised).

## What changed

1. **Table encoding (offline re-aggregation of the shipped matcher responses).**
   A category match is expanded to every catalog row of its kind and encoded at
   c/N per row with a -12 clip floor, so the kind collectively carries c while
   instance claims keep their weight. Built by `matching:reaggregate_tables
   --policy catexpand_divided` (ported from Harel's scratch `reaggregate.py`,
   verified identical on all four artifacts) and injected with the new
   `--tables_override` flag; the same rule is implemented in the
   `category_chunks` matcher layout on Harel's `harel/loc-experiments` branch
   (not in this PR; matching was not re-run).
2. **One identity vote per track**: Ethan's `--track_joint 1 --joint_slack 1`.
3. **Smoothing**: `--smoother fixed_interval --smooth_lag N --smooth_lags ...`
   adds a forward-backward pass over the same grid HMM (adjoint motion
   operator, checkpoint-and-recompute), scoring the end-of-run trajectory and
   online fixed-lag estimates; the causal score is unchanged.
4. `--likelihood_cache_gb`: per-epoch likelihood cache, bit-identical, ~3.6x
   faster replay.

## Results (divided tables + joint factor)

| sequence | shipped causal | new causal (online) | fixed-lag 30 (delayed 30 kf) | end-of-run smoothed (hindsight) |
|---|---|---|---|---|
| portland_flight_20260906_leg1 | 0.036 | 0.262 | 0.417 | 0.800 |
| flevoland_polder | 0.057 | 0.585 | 0.641 | 0.991 |
| boston_harbor_leg1 | 0.638 | 0.581 | 0.622 | 0.707 |
| mount_washington_20260815_leg2 | 0.864 | 0.866 | 0.903 | 1.000 |

The metric is the causal column. The `smooth_base` arm (shipped tables +
independent-epoch smoother) was deleted on 2026-09-12: its forward pass
applied each bearing at its anchor before the track had closed, so its
fixed-lag numbers used not-yet-released tracks. `grid_filter` now refuses
`--smooth_lag` without `--track_joint 1`. Fixed-lag numbers above are from
the joint arm and are legitimate but delayed; end-of-run smoothed is
hindsight only.

Portland online lag curve: lag 10/30/60/120/240 = 0.313/0.417/0.547/0.725/0.768.
Boston regresses at the end of the run (a confidently wrong lighthouse
instance released at the last keyframe becomes decisive under the joint
factor); fix candidates (`divided2`, `split` tables, `sum_conf` identity
share) were queued at the time of this note.

Privileged diagnostics (not evaluations): Portland labeled-truth oracle 0.681,
triangulation oracle 0.398 (Portland) / 0.841 (Flevoland); continuous 30 deg
truth-course heading + divided tables 0.449 (Portland).

## Figures (in the results directory)

- `overhead_online_<dataset>.png`: shipped causal | new causal (online, lag 0)
  | end-of-run smoothed (NOT online). Online lock: Portland kf 59, Flevoland
  kf 271, Boston kf 70.
- `overhead_before_after_<dataset>.png`: shipped causal vs new END-OF-RUN
  smoothed (hindsight; kept for the trajectory-recovery story).
- `traj_<dataset>.png`, `traj_smoothed_<dataset>.png`: truth mass and MAP
  error per keyframe.
- `lagcurve_portland_flight_20260906_leg1.png`: online score vs latency.

## Matcher

No matching was re-run for these results. Harel's `harel/loc-experiments`
branch also carries a `matching.set2_layout: category_chunks` layout (Set 2
grouped by kind, `category_matches` + `in_map_confidence` in the schema, kind
endorsements expanded catalog-wide at c/N) with sealed, unsubmitted request
sets for Portland and Flevoland. It is not part of this PR.
