# Farfield reorganization plan

This file is the working plan for migrating the far-field cross-view
geolocalization work from the `refactor-farfield` checkpoint branch into a
sequence of reviewable, stacked PRs off `main`. It is the anchor document for
a long migration: **update the status table at the bottom as PRs land**, and
record any decision that changes this plan in the decisions log.

Context: the project extends LOCI (loci-release) to far-field landmarks —
extract distant landmarks from panoramas/video, correspond them with OSM/ENC
landmarks, and do bearing-only localization. It must work across environment
types (harbors, mountains, landmark-sparse driving). The 2026-08-20 audit of
the checkpoint branch found systemic convention drift, missing provenance, a
disconnected pipeline tail, and duplicated/dead code; this migration fixes
those structurally rather than patching them in place.

## Ground rules (every PR obeys these)

1. **One owner per convention.** The camera frame, mount-offset semantics,
   angle helpers, ENU, haversine, earth constants each have exactly one
   definition, in `farfield/geometry.py`. Nothing restates them — not in
   code, not in docstrings, not in docs. Convention *strings* are exported
   constants that artifacts embed verbatim.
2. **No stale defaults.** Any argument whose value encodes a modeling or
   dataset assumption (model name, catalog version, thresholds, offsets,
   resolutions, ranges) is **required** — either on an explicit source/tool
   invocation or in the build config. Only mechanical knobs (worker counts,
   ports, verbosity) may
   default. If you find yourself wanting a default "so the command is
   shorter", the value belongs in the build config.
3. **Build config, not flag soup.** Result-shaping parameters are set once at
   build creation and recorded in `<build_dir>/build_config.json`. Subsequent
   stages take `--build_dir`, validate the record, and refuse scientific CLI
   overrides. Changing a parameter means a new build/artifact version unless
   a reviewed stage-reuse proof authorizes the exact unaffected prefix.
4. **Typed provenance on every scientific write.** `farfield/artifact.py`
   transactionally publishes `farfield.artifact.v1`: exact declared outputs,
   content digests, resolved config/build identity, producing commit, and
   typed upstream refs. Partial `.incomplete` state is never completion.
5. **Presentation is a sidecar, not a scientific mutation.** Viewer products
   bind exact scientific inputs in their own typed sidecar. Indexes attach a
   compatible sidecar to those exact versions, never fall back to obsolete
   embedded pages, and may be regenerated without changing scientific bytes.
6. **Scientific outputs use the data root.** Localization runs live in
   `runs/<experiment>/`; per-dataset scientific products live in
   `artifacts/`; mutable orchestration records live in `builds/`. Temporary
   transaction/work directories are permitted only as bounded implementation
   state and cannot become published provenance.
7. **Frozen means frozen.** Stages never mutate `datasets/`. Diagnostic
   calibration evidence is not authority; an approved dataset input is
   created only through its explicit review/finalize boundary and is then
   content-bound by every build that uses it.
8. **Docs carry no values.** Docs explain contracts and point at configs,
   `--help`, and code identifiers. Any literal default written into a doc is
   assumed stale on arrival.
9. **Delete, don't deprecate.** Superseded code is removed in the PR that
   supersedes it. The `refactor-farfield` branch remains as the reference for
   anything not ported.

## Target layout

```
experimental/overhead_matching/swag/farfield/
  geometry.py            # THE camera frame, angle arithmetic, earth, ENU
  paths.py               # data-root/lane/dataset resolution
  build_config.py        # immutable scientific build recipes
  artifact.py            # typed refs, validation, transactional publication
  stage_reuse.py         # reviewed exact prefix-reuse proof and authorization
  dataset.py             # dataset contract: frames, metadata, ingest
  audit_dataset.py       # the contract audit CLI
  testing.py             # synthetic dataset fixtures
  catalog/               # landmark schema reader + OSM/ENC catalog
  extraction/            # pinhole render, prompts, extraction stage, LLM cost
  tracking/              # SAM2 tracking, keyframes, semantic audit, tracklets
  calibration/           # alignment diagnostics and review evidence
  matching/              # whole-map uncertainty-preserving matching
  localization/          # export builder, filter, drivers, forensics, viewers
  pipeline.py            # orchestrator (new-build / run / status / prefix reuse)
  configs/               # explicit build recipes
  viewers/               # shared page helper + disk-scanned index chain
  collection/            # mapillary collection pipeline
  dataset_tools/         # ingest, trims, catalog sources, maintenance tools
docs/farfield/           # the docs, pipeline order, value-free
```

Existing `swag/landmark_filtering/`, `swag/bearing_only_localization/`,
`swag/mapillary_tools/`, and the farfield pieces of `swag/scripts/` +
`swag/data/farfield_paths.py` exist only on the checkpoint branch and are
ported (with fixes) into this tree; Pipeline A (`run_filter_pipeline`,
`filter_run_viewer`, `tracking.py`, `triangulation.py`, `yaw_offset.py`,
`heuristic_filters.py`, `semantic_similarity.py`, `artifact_schema.py`,
`configs/default.yaml`) is **not ported** (decision #1).

## Key design changes

### The merge stage is eliminated

`m6_merge_tracks` + `track_merge.py` (geometric track consolidation into
merged landmarks, plus a materialized `merged/{landmarks,pair_stats,
measurements}.json`) are deleted, not ported. Rationale: its own contract
says under-merging is cheap (two tracklets of one object both match the same
map feature; the filter's data association copes) while over-merging is
poison — and the audit found its consumers already disagree with it
(`m11` re-fuses with a different kappa rule; `m9` uses a different support
bar; three copies of `epoch_keyframes`).

Replacement: `tracking/tracklets.py`, a library (no stage, no artifact) that
computes localization-ready tracklets directly from `tracks_*.json` + the
semantic audit:

- the support gate (`min_supports`, from run config — a track that was never
  audited has no canonical semantics and must not reach matching or the
  filter);
- `bearing_series` (mask centroid → camera-frame azimuth via
  `geometry.azimuth_of_pano_column`, restricted to audit valid segments);
- `fuse_bearings` (per-epoch circular mean, width-aware kappa).

Offset sweep, matching, and export all call this one library with the run's
recorded fusion params. Each track is its own tracklet/matching unit;
duplicate tracks of one physical object are tolerated by design. The
parent/child, handoff, ambiguity-adjudication, and conflict machinery is
deleted. If over-splitting ever proves harmful, the fix belongs in the
filter's association model, not in a pipeline weld.

### Build config replaces flag defaults

`pipeline.py new-build --dataset <name> --config <yaml>` validates the exact
config schema and frozen inputs, then writes `build_config.json` under the
dataset/build orchestration lane. Stages are selected with
`pipeline.py run --build_dir <dir> [--from/--to/--only/--skip]`; scientific
settings come from the record. The pinned order is extraction → tracking →
semantic audit → bearing observations → matching → alignment diagnostics →
localization inputs → localization. Viewers and index refresh are derived
presentation work, not scientific stages.

A changed downstream input need not force extraction/tracking regeneration.
`prove-stage-reuse` creates a reviewed `stage_reuse.json` that revalidates the
source/target builds, protected inputs, producer contracts, exact artifact
refs, and prefix-code compatibility through tracking. Every downstream
consumer verifies the proof independently. Reuse never aliases, copies, or
re-stamps the old scientific artifacts.

### Provenance and versions

- `artifact.py` owns the scientific artifact schema, typed `ArtifactRef`,
  strict reopen/validation, and no-clobber transaction. Manifests name every
  output byte and exact upstream manifest/content identity.
- Build identity and stage orchestration digests bind the result-shaping
  config/input projection. Versions remain explicit; no reader resolves a
  default or “latest” scientific input.
- Consumers resolve their configured lane, reopen its manifest/payload, and
  compare exact refs. A manifest is evidence, not a substitute for validating
  the bytes a consumer actually uses.
- Provider stages use immutable request sets plus retained attempt shards.
  Canonical result artifacts publish only with complete, unique, valid key
  coverage; provider inconsistencies remain visible in ledgers.

### Nominal forward and frames (carried fixes)

- `geometry.py` exports the canonical camera-frame string; azimuth zero is
  the panorama centre column. Dataset ingest refuses north-aligned, rotated,
  or unqualified camera-frame imagery.
- `nominal_forward.py` owns the fixed camera-to-platform-forward calibration.
  It is not GPS course, and diagnostic sweeps/sun checks cannot grant it
  authority. Only an approved, dataset-bound record may rotate bearings.
- Bearing observations include only supported records inside canonical audit
  segments. Localization-input publication validates the calibration bytes,
  frame, target dataset, and exact bearing/match/catalog lineage.

### Catalog regeneration and retained extraction

Active catalog construction is report-first and digest-gated. It binds frozen
trajectory tables, caller-pinned complete OSM/ENC sources, coverage, audited
geometry repairs, and a typed full catalog. Spatial trimming is a separate
typed artifact with a strict reviewed plan; policy values live in that plan.

Previously paid extraction evidence may be adopted only through the explicit
legacy-adoption verifier. It reconstructs current pinhole/request bytes,
validates enumerated primary/retry history, records normalization/sanitation,
and publishes typed REQUEST → RESULT plus PINHOLE + RESULT → FRAME lineage
without a provider call.

### Audit, bearings, matching, and localization

Semantic audit uses one provider-facing decision discriminator that maps
deterministically to the stable canonical audit schema. Controlled drops and
invalid responses are counted, while systematic error rates remain a reason
to revise the contract. The tracker records support and propagation horizons
separately; accepted audit segments are clipped to supported evidence before
bearings are fused.

Machine-only matching preserves ambiguous candidates and a null hypothesis
instead of claiming human-confirmed identity. Human review is a separate
assisted lineage. Localization consumes typed bearing/match/catalog inputs,
uses a uniform prior for the primary evaluation, and supports recorded NumPy
or CUDA-backed Torch measurement evaluation. Viewer HTML is published as a
derived sidecar and never participates in scientific hashes.

### Data-root maintenance

The current layout is contract-owned by `paths.py`: frozen datasets,
scientific artifact lanes, build orchestration records, localization runs,
retained raw material, and an explicit archive. Maintenance begins with a
read-only inventory and exact reference graph. Superseded scientific
artifacts may be moved to a dated archive only after proving that no current
build, artifact, run, or viewer sidecar references them. Runs remain in
`runs/`; frozen datasets are not reorganized as pipeline cleanup.

Generated `index.html` files are disposable navigation. Refreshing indexes or
viewer sidecars does not authorize changing a scientific manifest. Obsolete
embedded track pages are neither linked nor accepted as the current viewer
standard.

## PR stack

Branch names `reorg/NN-<slug>` were stacked on the previous branch. The table
is a historical record of the original split; the current contracts are the
sections above and the follow-on regeneration implementation, not stale
interfaces named in an earlier PR description.

| PR | branch | contents | state |
|----|--------|----------|-------|
| 00 | `reorg/00-plan` | This file. | landed |
| 01 | `reorg/01-geometry` | `farfield/geometry.py`: the single owner for camera frame, mount offset, angle/ENU/earth helpers, body↔world bearings. Stale KNOWN-ISSUE prose gone; one off-axis-correct elevation; the one haversine. | landed |
| 02 | `reorg/02-paths-provenance` | `paths.py` (no default version/catalog/checkpoint), `provenance.py` (the one manifest writer + byte-identical-version refusal), `run_config.py` (immutable recorded runs). | landed |
| 03 | `reorg/03-dataset` | `dataset.py` contract + ingest; `north_aligned` and mount-offset-qualifier gates; `audit_dataset` with real argparse. | landed |
| 04 | `reorg/04-catalog` | `catalog/schema.py` (sole feather reader) + `catalog/catalog.py` (ENU catalog, far-field vocabulary, extents, cache). | landed |
| 05 | `reorg/05-tracklets` | `tracking/tracklets.py` — **the merge-stage replacement**, with fusion parity pinned against the old m6 output. | landed |
| 06 | `reorg/06-localization-contract` | `localization/{structs,filter_catalog,run_io,export_ingest,gps_to_odometry}` — schema 0.3 with required provenance; offset-frame enforcement at the export boundary. | landed |
| 07 | `reorg/07-extraction` | `extraction/{llm_cost,vertex_batch_manager}` — one execution-flag block, `--model` required everywhere. | landed |
| 08 | `reorg/08-viewers` | `viewers/page.py` (one stylesheet + provenance footer) and `viewers/indexes.py` (disk-scanned index chain to the data root). | landed |
| 09 | `reorg/09-calibration` | Sun check (verdict-gated `usable`), offset sweep, heading model, `audit_io` — sidecars only, no metadata writes. | landed |
| 10 | `reorg/10-tracking` | SAM2 tracking stage + keyframe viewer; crash-safe `tracks_complete.json`; readers use recorded config. | landed |
| 11 | `reorg/11-matching` | Whole-map LLM matching + match viewer, per-track Set 1; msgspec-typed `compatibility.json`; feather from recorded settings. | landed |
| 12 | `reorg/12-filter` | The particle filter core: resection, mode tracker, proposal, filter, torch backend, scenario, `metrics.py` split out. | landed |
| 13 | `reorg/13-run-drivers` | `build_export` (one tool, replacing both m11 halves), `run_export`, `run_localization`. | landed |
| 14 | `reorg/14-extraction-stage` | `panorama_to_pinhole`, the torch-free prompt registry, and the extraction stage orchestrator. | landed |
| 15 | `reorg/15-shared-schema` | The dict-tags feather schema adopted in the shared tree (`common/openstreetmap`, `swag/data`, `swag/model`, 4 scripts). | landed |
| 16 | `reorg/16-audit-stage` | Semantic audit library + `audit_requests` / `audit_review`; settings.json provenance; recorded-config readers. | landed |
| 17 | `reorg/17-forensics-viewers` | Basemap, satellite underlay, plots, the self-contained viewer (+ server), forensics/replay/attribution and the `forensics` CLI. | landed |
| 18 | `reorg/18-collection` | Mapillary collection: workspace-relative root, coverage gate importable, `mount_offset_frame` written, relics dropped. | landed |
| 19 | `reorg/19-dataset-tools` | Self-collect ingest, trims, ENC/catalog building, and `publish_mount_offset` — the one guarded metadata writer. | landed |
| 20 | `reorg/20-pipeline` | `pipeline.py` (new-run / run / status) through `viewer`, run configs, index refresh after every stage. | landed |
| 21 | `reorg/21-docs` | `docs/farfield/`: README, conventions, datasets, pipeline, localization — value-free, current-state, pipeline order. | this PR |

## Decisions log

- 2026-08-20 (ekf): Pipeline A is not ported (archive via the checkpoint
  branch). Filter runs live only in `runs/<experiment>/`; matching + exports
  stay with their tracking run. `archive/bad_trajectories` untouched for
  now. Runbook: full rewrite, not patched.
- 2026-08-20 (ekf): the merge stage is eliminated entirely (design above).
- 2026-08-20 (ekf): no defaults on assumption-carrying args; viewers/docs
  must auto-update rather than embed soon-stale values.

## Status

The original reorganization stack is complete. Follow-on regeneration work
now supplies the strict typed artifact chain, catalog
materialization/coverage, retained-evidence adoption, stage-scoped prefix
reuse, provider lifecycle, bearing artifacts, Torch localization, and viewer
sidecars described above. Legacy artifacts and runs remain historical
regression evidence only; they cannot be silently adopted into the current
typed chain.
