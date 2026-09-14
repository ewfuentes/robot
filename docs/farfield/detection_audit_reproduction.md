# Reproducing detection-first, audited-track localization

This change is based on `farfield-crossview-base`, which includes merged
#725 (distance windows) and #724 (the corrected grid filter).
The hybrid implementation is the version used
for the 35-window seed-0 experiment and the subsequent five-seed sweep.
The scientific `grid_filter.py` and `detection_audit.py` files are unchanged
from that experiment. No deduplication, range-bin change, or final smoothing
is introduced here.

## Evidence policy

Every matched single detection enters at its own keyframe. When an accepted
track reaches its existing natural release, withdraw its source-support raw
factors and insert its audited joint factor. Restore the checkpoint before
the earliest withdrawn factor and replay forward to the current keyframe.
Previously emitted scores remain immutable. This is not division by old
likelihoods at the current marginal, and is not final/backward smoothing.

The track-support map uses the canonical positive support classes, including
birth observations, and excludes bystanders. Shared claims remove a raw
factor only once. Unclaimed raw detections remain, including those from
rejected or boundary-crossing tracks. Partial audits replace all source
supports with the audit-valid track geometry. Consequently, this changes
both evidence timing and evidence retention; it is not a pure timing ablation.

Replay is exact for the selected pose-grid factor graph, not for the full
historical-pose posterior: the existing joint-track likelihood still expresses
track geometry at its release pose. Cached frontend latency and calibration
are not certified online. The earlier oracle-eager probe reused eventual
semantics early and must not be confused with this policy.

## Frozen inputs and settings

Use the archived experiment bundle under the shared `farfield_matching` data
root. Raw images, credentials, host-specific launch manifests, and result
payloads are not committed to Git.

| Run directory | Contents |
| --- | --- |
| `260914_distance_episodes` | 35-window delayed-tracking and no-tracking references; exact episode plans, natural-release schedules and divided matching tables |
| `260914_detection_audit_replay` | 35 hybrid seed-0 outputs, source-member maps, configurations, full report and read-only validation record |
| `260914_detection_audit_seedcheck` | Selected Charles/Flevoland seeds 1/2, all three arms |
| `260914_eager_diagnostic` | Earlier oracle-semantic timing probe, not the hybrid |
| `260914_audit_retention` | Separate removal-only and rejection controls, with their separately archived extended source; not enabled by this PR |
| `260914_hybrid_notracking_5seeds` | Five-seed hybrid/no-tracking sweep manifest, results and status |

The workers need the paired `artifacts/localization_inputs` directories and
the frozen plans, schedules, tables, and source-member maps. They need not
rerun detection, tracking, audits, matching, or model API requests. The
overnight handoff manifest records checksums for the results and inputs.

All scientific CLI settings come from each reference result's `config`, not
from current defaults. The reference configuration has 100 m cells, 36
heading bins, the calibrated Epson planar IMU model, range bins enabled,
the sum mixture, non-deduplicated source tracks, and checkpoints every eight
keyframes. It uses `track_joint=1`, `joint_backend=fused`, `joint_chunk=128`,
`likelihood_cache_gb=0`, `smoother=none`, `smooth_lag=0`, and empty `smooth_lags`.
Keep every other likelihood/motion parameter from the stored configuration.

The reproduction commands below retain the archived 35-window roster. The
[revised evaluation protocol](distance_episode_evaluation.md) instead specifies
five subtracks and a separate full trajectory for each of the 13 recordings;
do not relabel historical results as that revised study. Both protocols use
the same boundary policy. Each window resets the parent-region prior and IMU error;
an unsplit full leg retains its original seed stream. Use paired seeds
0–4 across both methods. No reverse traversal or artificial boundary flush.

## Build and reproduce one result

```bash
bazel build //experimental/overhead_matching/swag/farfield/localization:reproduce_grid_result
bazel test //experimental/overhead_matching/swag/farfield/localization:grid_filter_test \
  //experimental/overhead_matching/swag/farfield/localization:distance_episodes_test \
  //experimental/overhead_matching/swag/farfield/localization:reproduce_grid_result_test

bazel-bin/experimental/overhead_matching/swag/farfield/localization/reproduce_grid_result \
  --reference /data/farfield_matching/runs/260914_detection_audit_replay/results/local/charles_river_20260727.hybrid.episode0.seed0.causal.json \
  --out /tmp/charles.hybrid.seed0.reproduced.json
```

Add `--odometry-seed 1` to draw a different paired IMU realization. If moving
the bundle, add `--path-map /data/farfield_matching /new/data/root`. Historical
references can also contain the equivalent root
`/home/ekf/farfield_tracking_batch2_20260913`; map that root too. Path mappings
change only directory prefixes in the invocation, not input-file contents,
artifact hashes, or scientific settings.

The reproducer preserves completed/partial outputs, rejects smoothing, checks
the grid/episode/configuration and complete finite causal mass series, and
records reference/source hashes. For an unchanged seed it also checks every
mass value against the reference (absolute tolerance 2e-4 for cross-device
roundoff). Use a new output path for each attempt. It is one process per
evaluation; no cluster scheduler or machine assignments are embedded.

## Reproduce all 35 pairs, five seeds

Build once, then run the following stdlib-only driver from the worktree root.
Use an empty output directory. This runs sequentially; for multiple workers,
partition the reference list into disjoint subsets. It deliberately does not
alter other machines' jobs or assume that an SSH alias names the local host.

```python
from pathlib import Path
import subprocess

data = Path('/data/farfield_matching')
hybrid = data / 'runs/260914_detection_audit_replay/results'
baseline = data / 'runs/260914_distance_episodes/results'
output = Path('/tmp/hybrid-reproduction')
binary = Path('bazel-bin/experimental/overhead_matching/swag/farfield/localization/reproduce_grid_result').resolve()
references = sorted(hybrid.glob('*/*.hybrid.episode*.seed0.causal.json'))
assert len(references) == 35
for tracked in references:
    raw_name = tracked.name.replace('.hybrid.', '.no_tracking.')
    raw = list(baseline.glob('*/' + raw_name))
    assert len(raw) == 1
    for seed in range(5):
        for reference in (raw[0], tracked):
            destination = output / reference.name.replace('.seed0.', f'.seed{seed}.')
            subprocess.run([
                str(binary), '--reference', str(reference), '--out', str(destination),
                '--odometry-seed', str(seed),
                '--path-map', '/data/farfield_matching', str(data),
                '--path-map', '/home/ekf/farfield_tracking_batch2_20260913', str(data),
            ], check=True)
```

For the original one-seed delayed-track comparison, pass the matching
`.current.episodeN.seed0.causal.json` reference to the same reproducer.
The experimental `detections_only` policy and `output_end` prefix controls
are also retained in their stored configurations. Separate retention-control
variants have their exact source snapshots in their own run directory; this
PR does not silently adopt their changed evidence policy.

## Rebuild a source-member map when needed

Normally reuse the checksummed map in `260914_detection_audit_replay/plans`.
If reconstructing it, also provide the source tracking artifacts bound by
the two episode plans:

```bash
bazel run //experimental/overhead_matching/swag/farfield/localization:build_detection_audit_plan -- \
  --tracked_input /path/to/tracked/localization_inputs \
  --detection_input /path/to/single_detection/localization_inputs \
  --tracked_episode_plan /path/to/tracked_episode_plan.json \
  --detection_episode_plan /path/to/detection_episode_plan.json \
  --artifacts_dir /data/farfield_matching/artifacts \
  --out /tmp/new_source_member_map.json
```

The builder verifies paired truth, map and clean motion and checks source
artifact identities. Its map records missing observations and binds both
localization inputs. Never substitute final-audit semantics into detections
before natural release.

## Scoring and interpretation

Report `summary.dn_mass_500` and `summary.dn_mass_100`. At each keyframe score
only the current posterior against the true current position, then integrate
that emitted mass series along traveled distance. These are causal scores,
not scores of a final smoothed path. Keep per-window/per-seed comparisons;
average seeds/windows within each parent before an equal-parent summary.
Overlapping windows and synthetic seeds are not independent recordings.

The seed-0 equal-parent mass@500 / mass@100 values were 28.52 / 7.14% for
delayed tracking, 24.39 / 9.42% for no tracking, and 37.18 / 12.54% for the
hybrid. All 13 parents and all 35 windows are included, including regressions.
Use the full archived report for per-window values and qualifications;
these window results are not replacements for the original full-leg table.
