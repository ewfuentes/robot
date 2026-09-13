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
   instance claims keep their weight. Built by `reaggregate.py` (policy
   `catexpand_divided`) and injected with the new `--tables_override` flag; the
   same rule is implemented in the matcher's new `category_chunks` layout.
2. **One identity vote per track**: Ethan's `--track_joint 1 --joint_slack 1`.
3. **Smoothing**: `--smoother fixed_interval --smooth_lag N --smooth_lags ...`
   adds a forward-backward pass over the same grid HMM (adjoint motion
   operator, checkpoint-and-recompute), scoring the end-of-run trajectory and
   online fixed-lag estimates; the causal score is unchanged.
4. `--likelihood_cache_gb`: per-epoch likelihood cache, bit-identical, ~3.6x
   faster replay.

## Results (divided tables + joint factor)

| sequence | shipped causal | new causal (online) | fixed-lag 30 (online) | end-of-run smoothed | shipped smoothed |
|---|---|---|---|---|---|
| portland_flight_20260906_leg1 | 0.036 | 0.262 | 0.417 | 0.800 | 0.117 |
| flevoland_polder | 0.057 | 0.585 | 0.641 | 0.991 | 0.176 |
| boston_harbor_leg1 | 0.638 | 0.581 | 0.622 | 0.707 | 0.877 |
| mount_washington_20260815_leg2 | 0.864 | 0.866 | 0.903 | 1.000 | 1.000 |

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

`matching.set2_layout: category_chunks` (new config key, default
`digest_chunks` byte-stable): Set 2 grouped by kind, `category_matches` +
`in_map_confidence` in the response schema, kind endorsements expanded
catalog-wide and encoded at c/N. Request sets sealed, unsubmitted:
`artifacts/landmark_matches/{portland_flight_20260906_leg1,flevoland_polder}/catmatch_20260912_v1.llm-work`.

## Addendum (later on 2026-09-12)

- `--smooth_lags 10,60,120` scores several online fixed lags in one backward
  chain and records each lag's MAP positions (`smoothing.fixed_lags[L]`).
  Portland (divided + joint): lag 10/30/60/120/240 = 0.313/0.417/0.547/0.725/0.768,
  end of run 0.800. Flevoland: lag 10/30/60/120 = 0.605/0.641/0.690/0.786, end 0.991.
- `--identity_share sum_conf` (endorsed mass from summed row confidences):
  Portland 0.173, Flevoland 0.575, Boston 0.623, MtWash 0.865 — confidences are
  not calibrated enough for this to help uniformly.
- `--eof_temper`: tempering only end-of-run flush releases changes nothing causal.
- Zero-latency bound (eager, divided tables, independent epochs): Portland 0.196
  vs natural 0.120; Flevoland 0.155 vs 0.115.
- Joint factor bounds: `--joint_temper 0.5` Boston 0.710/0.976 but Portland
  0.127/0.470; `--joint_cap 20` Portland 0.365/0.945, Boston 0.628/0.912,
  Flevoland 0.506/0.981 (cap sweep continuing).
- Smoother: per-keyframe factor products kept in a rolling host window so
  fixed-lag chains do not rebuild them; likelihood cache unpinned.

## Joint-cap sweep and candidate default lane

`--joint_cap` (clamp a track's mixture term before null and tail) with divided
tables + `--track_joint 1 --joint_slack 1`, causal / lag30 / smoothed, seed 0:

| cap | Portland | Flevoland | Boston | Mt Washington |
|---|---|---|---|---|
| none | 0.262 / 0.417 / 0.800 | 0.585 / 0.641 / 0.991 | 0.581 / 0.622 / 0.707 | 0.866 / 0.903 / 1.000 |
| 100 | 0.352 / 0.419 / 0.946 | 0.552 / 0.577 / 0.990 | 0.640 / 0.691 / 0.911 | 0.862 / 0.900 / 0.9995 |
| 50 | 0.355 / 0.422 / 0.946 | 0.531 / 0.556 / 0.987 | 0.632 / 0.682 / 0.918 | - |
| 20 | 0.365 / 0.432 / 0.945 | 0.506 / 0.531 / 0.981 | 0.628 / 0.675 / 0.912 | 0.848 / 0.887 / 0.9995 |

Candidate default lane: divided tables + joint factor + `--joint_cap 100`. Versus
the shipped pipeline (causal): Portland 0.036 -> 0.352, Flevoland 0.057 -> 0.552,
Boston 0.638 -> 0.640, Mt Washington 0.864 -> 0.862; end-of-run smoothed
0.117 -> 0.946, 0.176 -> 0.990, 0.877 -> 0.911, 1.000 -> 0.9995. Single seed.
Full detail: `/data/farfield_matching/runs/260912_harel_experiments/HANDOFF.md`.
