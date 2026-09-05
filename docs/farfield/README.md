# Far-field cross-view geolocalization

Farfield extends LOCI to distant landmarks: extract landmarks from panorama
sequences, track them through time, relate the accepted evidence to OSM/ENC
map landmarks, and localize an agent from a uniform prior with bearing-only
updates. The pipeline is intended to use the same scientific contract across
harbors, mountains, and other landmark-sparse environments.

All code lives under `experimental/overhead_matching/swag/farfield/`:

| package | what it owns |
|---|---|
| `geometry.py`, `nominal_forward.py` | frames, angle/ENU helpers, and the approved camera-to-platform-forward calibration |
| `paths.py`, `build_config.py`, `artifact.py` | data-root layout, immutable build recipes, typed artifact publication and validation |
| `artifact_identity.py`, `code_provenance.py` | what determines an artifact, hashed per stage; and the code that made it, recorded but never gating |
| `artifact_recipe.py` | the settings, inputs and lineage an artifact records so it can be reproduced and checked without a build directory |
| `dataset.py`, `audit_dataset.py` | the frozen dataset contract and its validation |
| `catalog/`, `collection/`, `dataset_tools/` | catalog schema, source collection, full-catalog materialization, coverage, and trims |
| `extraction/` | pinhole rendering, VLM requests, provider lifecycle, and retained-evidence adoption |
| `tracking/` | object tracking, semantic audit, accepted tracklets, bearings, and frame/track viewer sidecars |
| `matching/` | whole-catalog candidate generation and uncertainty-preserving matching |
| `calibration/` | alignment diagnostics; diagnostic estimates never become calibration authority implicitly |
| `localization/` | typed localization inputs, particle filter, NumPy/Torch backends, forensics, and run viewers |
| `pipeline.py` | the scientific orchestrator (`new-build`, `run`, `status`, and reviewed prefix reuse) |
| `viewers/` | generated data-root indexes and shared page infrastructure |

Read in pipeline order:

1. [`conventions.md`](conventions.md) — frames, signs, zero points, and the
   incidents that make the rules non-negotiable. **Read first.**
2. [`datasets.md`](datasets.md) — frozen datasets, catalog construction, and
   the data-root layout.
3. [`selfcollect.md`](selfcollect.md) — GPS/video synchronization,
   anonymization, and the human privacy-review gate for collected video.
4. [`pipeline.md`](pipeline.md) — immutable builds, typed artifacts, the eight
   scientific stages, provider recovery, and safe prefix reuse.
5. [`localization.md`](localization.md) — the bearing seam, machine matching,
   filter backends, evaluation rules, and derived viewers.
5. [`loci_pipeline.md`](loci_pipeline.md) — the separate released-LOCI
   late-fusion baseline, from prepared inputs through evaluation.

## Ground rules

- **No assumption-carrying defaults.** Model names, artifact versions,
  thresholds, calibrations, resolutions, and ranges are explicit in the
  build config. Mechanical controls may remain command-line options.
- **Build recipes are immutable; artifacts are independently immutable.**
  `pipeline new-build` seals every scientific input and setting in
  `build_config.json`. Scientific outputs live in versioned artifact lanes,
  not in the build directory.
- **Publication is typed and transactional.** Every completed artifact has a
  `farfield.artifact.v1` manifest, exact declared outputs, content digests,
  and typed upstream references. Publication uses an `.incomplete` sibling
  and never treats partial output as complete.
- **Reuse is stage-scoped, not all-or-nothing.** An artifact is identified
  by its own stage's resolved config, its upstreams' manifest digests, and
  the build inputs that stage reads. A change downstream of tracking does not
  move the identity of the extraction and tracking above it, so those are
  reused because they still match -- no authorization, attestation or bridge
  is involved.
- **Datasets are frozen.** Pipeline stages never mutate `datasets/`. Approved
  nominal-forward records and other dataset inputs are content-bound before
  use.
- **Scientific and presentation products are separate.** Viewer HTML is a
  derived sidecar. Viewer-only changes publish a new sidecar and never change
  or invalidate the scientific artifact it presents.
- **Docs carry no tuned values.** Result-shaping values live in reviewed
  configs and plans; command interfaces live in each target's `--help`.

## Browsing results

Serve the data root over HTTP and open `artifacts/` or `runs/`. Generated
indexes attach the newest valid viewer sidecar to the exact frame/track
artifact pair it presents. They never fall back to obsolete HTML embedded in
old track artifacts; a missing sidecar is shown explicitly. Catalog coverage
is a separate review artifact rather than a scientific pipeline stage.

After scientific tracking publishes, the launcher attempts a frame-landmark
viewer sidecar containing annotated keyframes and linked derived track pages.
A viewer failure cannot roll back or contaminate the completed track artifact.
Videos and thumbnails are referenced from that exact artifact instead of being
copied. Regenerate navigation after filesystem maintenance with
`farfield/viewers:refresh_indexes`.
