# LOCI aggregation ablations

Updated 2026-09-29: recalibrate landmark sigma for every setting first. Fixed-sigma
reruns and sigma sensitivity studies are deferred. No training is needed.

## Settings

Change one factor at a time from the shared baseline: Hungarian, dustbin enabled,
threshold 0.8, inverse-log uniqueness weights, probability sum, normalized residual.
Uniqueness weighting is applied after assignment and counts all archived OSM
columns, including landmarks outside the current patch grid.

| Setting | Export flags differing from baseline | Residual |
| --- | --- | --- |
| baseline | none | normalized |
| threshold_050 | `--prob_threshold 0.5` | normalized |
| threshold_065 | `--prob_threshold 0.65` | normalized |
| threshold_090 | `--prob_threshold 0.9` | normalized |
| threshold_095 | `--prob_threshold 0.95` | normalized |
| greedy | `--method greedy` | normalized |
| count | `--aggregation count` | normalized |
| max | `--aggregation max` | normalized |
| unweighted | omit `--uniqueness_weighted` | normalized |
| inverse_count | `--uniqueness_weighting inverse_count` | normalized |
| raw_residual | reuse baseline similarity matrix | raw |
| no_dustbin | `--no_dustbin` | normalized |

`count` sums the weights of accepted matches (or counts matches if unweighted).
`inverse_count` uses `1 / max(1, N_i)`; the default remains inverse-log.
Count/max change aggregation only, preserving the selected assignment. The
threshold sweep changes the dustbin value, acceptance cutoff, and uniqueness
counts together, matching deployed behavior. Independent best-OSM matching is
excluded; a threshold sweep with fixed uniqueness weights is deferred.

## Export, calibrate, evaluate

Use the existing tools with a different output directory per setting. Example
baseline export (change the table's flags for each alternative):

```sh
bazel run //experimental/overhead_matching/swag/scripts:export_correspondence_similarity -- \
  --from_raw "$DATASET/correspondence_scores/simple_v1_v6_raw.pt" \
  --dataset_path "$DATASET" --landmark_version "$LANDMARK_VERSION" \
  --output_path "$OUTPUT/baseline.pt" --compute_similarity \
  --method hungarian --aggregation sum --prob_threshold 0.8 \
  --uniqueness_weighted --workers 8
```

The exported matrix is `$OUTPUT/baseline_similarity.pt`. Multiple workers are
opt-in and use Linux fork to share the raw matrix; use a fresh CPU-only process
with `--from_raw`, not the model-inference path. Serial execution remains default.
Limit OpenMP/BLAS threads to one per worker to avoid CPU oversubscription.

For **each setting**, export its Seattle matrix and run `calibrate_sigma` with
`--residual-form normalized --exclude-zero-sim-true`. For `raw_residual`, reuse
Seattle's baseline matrix and use `--residual-form raw --exclude-zero-sim-true`.
Keep semipositives included and constant rows excluded for every setting. Check
that the fitted `sigma_mle_per_pair` is finite and positive before evaluating.
No Seattle path replay is required for calibration.

Use `SafaPlusNormalizedLandmarkAggregatorConfig` with image sigma **0.1809**,
the setting's full-precision Seattle-fitted landmark sigma, and
`landmark_use_raw_residual: true` only for the raw variant. Run the existing
`evaluate_histogram_on_paths` with the original paths, seed 42, motion noise
0.141, odometry noise 0.141/seed 7919, subdivision 4, and radii 25/50/100 m.
Record all export flags, input identities, fitted sigmas, and code revision.

## First batch and deployment

First complete Boston Snowy (`Boston`), Framingham, Fort Myers
(`post_hurricane_ian_sw`), San Francisco (`SanFrancisco_mapillary`), and Middletown.
Twelve settings x five conditions = **60 evaluations / 60,000 path replays**,
plus **12 Seattle calibration fits** (11 distinct Seattle matrices). Include a
fresh baseline in this budget; reuse only after checking inputs and fitted sigma.

Use pika, palm, and laura. Keep this host free of evaluation workloads. Stage
isolated checkouts and exact inputs after opening the PR; do not start the
experiment queue before review. One filter evaluation at a time per GPU.
The remaining five test conditions can follow later, bringing the complete
study to 120 evaluations / 168,000 path replays without duplicate fixed-sigma runs.
Use the corrected September 27 Netherlands image matrices for that later batch.

## Timing evidence

Completed LOCI logs under
`/data/overhead_matching/evaluation/results/260522_full_rerun_no_hinge/cosmos_no_hinge_sigma0.46/`
record approximately 14.8 min for Boston, 9 for Framingham, 12 for Fort Myers,
6.3 for SF, and 3.9 for Middletown: **46 minutes per five-condition pass**, or
**9.2 GPU-hours for 12 settings**, before rescoring, calibration, and startup.
September semantic-ablation logs corroborate these timings. Machine speeds
vary, so this is aggregate compute, not a guaranteed three-machine wall time.

Previous all-condition estimate: about 137 minutes of filtering per setting;
27.4 GPU-hours for all twelve settings. Matrix exports are additional CPU work.
Share exports between calibration/evaluation where possible; the raw-residual
variant always reuses the baseline matrix. Defer calibration-method changes,
classifier calibration, confidence gating, and other study families.
