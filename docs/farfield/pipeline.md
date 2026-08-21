# Running the pipeline

One dataset, end to end: frames → tracks → matches → localization. All
parameter values live in the run's recorded config — this document carries
none on purpose (a value printed here is stale the day after it is tuned).
The example configs in `farfield/configs/` are the reference values; each
tool's `--help` is the reference interface.

## The run-config model

1. **Create a run.** Copy an example config, edit it, and record it:

   ```
   bazel run //experimental/overhead_matching/swag/farfield:pipeline -- new-run \
       --dataset <name> --run_name <rNNN> --config <your.yaml>
   ```

   `new-run` validates that every result-shaping key is present (all missing
   keys reported at once) and writes `run_config.json` into the run
   directory. Runs are immutable: to change a value, start a new run.

2. **Execute stages.** Parameters come from the record; the CLI only selects
   which stages run:

   ```
   bazel run //...farfield:pipeline -- run --run_dir <run> [--from S] [--to S] [--only S] [--skip S] [--force]
   bazel run //...farfield:pipeline -- status --run_dir <run>
   ```

Model stages default to the Batch API; `--online` swaps the run to on-demand
calls (faster, roughly twice the price). What a run actually cost is measured
from the stored `usageMetadata`, never estimated.

## Stages, in order

| stage | what it does | output (also its completion marker) |
|---|---|---|
| `extract` | pinhole faces + VLM landmark detections | `frame_landmarks/<v>/` artifact + manifest |
| `track` | SAM2 tracking (GPU, the long pole) | `tracks_*.json` per range + `tracks_complete.json` |
| `keyframes` | per-keyframe detection pages | `keyframes/index.html` |
| `audit` | semantic audit of every supported track | `semantic_audit/results.jsonl` |
| `review` | audit review pages | `semantic_audit/review/index.html` |
| `offset` | sun check (absolute) then offset sweep (relative) | the two sidecar JSONs |
| `match` | whole-map LLM matching | `matching/matches.json` + `compatibility.json` |
| `matchview` | match review pages | `matching/review/index.html` |
| `export` | localization export (offset baked in here) | `localization_export/` |
| `localize` | the particle filter, uniform prior | a run dir under `runs/<experiment>/` |
| `plots` | map/strip/animation | `plots/` in the localization run |
| `viewer` | the self-contained run viewer | `viewer.html` in the localization run |

The ordering is pinned by `pipeline_test.StageOrderTest`, not by this table.
Every stage refreshes the index chain when it finishes, so the data root
stays fully browsable throughout.

## The two hard stops

- **Incomplete extraction.** Frames with no VLM response read downstream as
  frames containing no objects, so tracks crossing them starve. Checked
  before *any* detection-consuming stage; repair with the extract stage's
  `--retry_failed` (never `--force`, which re-bills the whole extraction).
- **No usable mount offset at export.** The export bakes the offset into
  every bearing. With no validated dataset record and no usable sidecar it
  refuses; see `conventions.md` §2 for the evidence ordering.

Everything else is advisory: it prints, records, and carries on. The sun
check abstaining (overcast) is a correct outcome and does not stop the run.

## Crash recovery

The tracking stage's completion marker is written per range *after* the range
finishes; a mid-run crash resumes the unfinished ranges instead of skipping
the stage. `status` shows every stage's marker. `--force` re-runs a stage
whose marker exists.

## Where things land

- Per-dataset stage products: `artifacts/<kind>/<dataset>/<version>/` —
  matching and exports live inside their tracking run.
- Localization runs: `runs/<experiment>/<dataset>_<run>/`. Every experiment
  directory carries an `experiment.md` (what is being explored, status,
  conclusions); its index page renders it.
- Nothing ever writes to `/tmp` or mutates `datasets/`.
