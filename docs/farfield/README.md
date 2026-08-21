# Far-field cross-view geolocalization

Extending LOCI to landmarks that are far away: extract distant landmarks from
panoramas/video with a VLM, track them over time, correspond them with OSM/ENC
map landmarks, and localize with bearing-only filtering. Target environments:
harbors, mountains, landmark-sparse driving.

All code lives under `experimental/overhead_matching/swag/farfield/`:

| package | what it owns |
|---|---|
| `geometry.py` | every frame/angle/ENU convention (the single owner) |
| `paths.py`, `provenance.py`, `run_config.py` | data-root layout, the one manifest writer, recorded run configs |
| `dataset.py`, `audit_dataset` | the dataset contract, its validation, detection ingest |
| `catalog/` | landmark-feather schema (the one reader) + map-side catalog |
| `extraction/` | VLM landmark extraction, LLM cost guard, Vertex batch manager |
| `tracking/` | SAM2 tracking, keyframe pages, semantic audit, tracklets |
| `calibration/` | mount-offset estimation (sun check, sweep) — sidecars only |
| `matching/` | whole-map LLM matching + review viewer |
| `localization/` | export builder, particle filter, run drivers, forensics, viewers |
| `pipeline.py` | the end-to-end orchestrator (`new-run` / `run` / `status`) |
| `viewers/` | the shared page helper + the disk-scanned index chain |
| `collection/` | Mapillary dataset collection |
| `dataset_tools/` | self-collect ingest, trims, catalog building, triage |

Read in pipeline order:

1. [`conventions.md`](conventions.md) — frames, signs, zero points, and the
   incidents that make the rules non-negotiable. **Read first.**
2. [`datasets.md`](datasets.md) — the dataset contract, how datasets are
   collected/ingested/audited, and the data-root layout.
3. [`pipeline.md`](pipeline.md) — running a dataset end to end with recorded
   run configs.
4. [`localization.md`](localization.md) — exports, filter runs, evaluation
   rules, and forensics.

## Ground rules (enforced in code, summarized here)

- **No assumption-carrying defaults.** Model names, catalog versions,
  thresholds, offsets, resolutions are required — on the CLI or in the run
  config. If a command feels long, the values belong in the config file, not
  in a default.
- **Runs are immutable records.** `pipeline new-run` validates and records
  every result-shaping value; stages read the record; changing a value means
  a new run. Readers and viewers always use the *recorded* config.
- **Every artifact names its inputs** (`provenance.py`: git commit, argv,
  inputs, config, timestamp). An artifact without a manifest is a bug.
- **Datasets are frozen.** No stage mutates `datasets/`; calibrations live in
  run-dir sidecars, and `dataset_tools:publish_mount_offset` is the one
  guarded writer of dataset metadata.
- **Everything is browsable.** Every stage writes its viewer and refreshes
  the index chain; `python -m http.server` at the data root reaches every
  page by clicking. Docs carry no literal parameter values — those live in
  `farfield/configs/` and each tool's `--help`.

## Browsing results

Serve the data root over HTTP and click: root → lanes → datasets/artifact
versions/experiments → stage pages. `runs/<experiment>/` directories carry an
`experiment.md` (what is being explored, status, conclusions) rendered on
their index page. Regenerate navigation manually with
`farfield/viewers:refresh_indexes` (stages do it automatically).
