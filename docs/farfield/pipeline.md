# Running the pipeline

The scientific chain is:

```
frozen dataset + catalog
  -> extraction -> tracking -> semantic audit -> bearings
  -> matching -> diagnostics -> localization inputs -> localization
```

All result-shaping values live in an immutable build config. This document
contains no tuned values; use the reviewed configs under `farfield/configs/`
and each target's `--help`.

## Immutable builds

Create a build from a reviewed YAML recipe:

```
bazel run //experimental/overhead_matching/swag/farfield:pipeline -- new-build \
    --dataset <dataset> \
    --farfield_root <root> \
    --build_name <build> \
    --config <recipe.yaml>
```

`new-build` rejects missing, unknown, or invalid config leaves and writes a
sealed `build_config.json` under `builds/<dataset>/<build>/`. The build
directory is mutable orchestration state only. Scientific outputs go to the
configured immutable artifact versions; the localization result goes under
`runs/`.

Run or inspect the build with:

```
bazel run //...farfield:pipeline -- run --build_dir <build-dir> [--from S] [--to S] [--only S] [--skip S]
bazel run //...farfield:pipeline -- status --build_dir <build-dir>
```

There is no `--force`. A completed artifact is reopened and fully validated;
a changed setting or output requires a new version/build. A failed producer
may resume only through its explicit lifecycle contract.

## Scientific stages

| stage | inputs | typed output |
|---|---|---|
| `extract` | frozen dataset, extraction recipe | `pinhole_images`, `frame_landmarks` |
| `track` | pinholes, frame detections | `object_tracks` |
| `audit` | tracks, frame detections | `semantic_audits` |
| `bearings` | tracks, canonical audit | `bearing_observations` |
| `match` | tracks, audit, catalog | `landmark_matches` |
| `diagnostics` | bearings, approved nominal-forward record | `alignment_diagnostics` |
| `localization_inputs` | bearings, matches, catalog, motion/calibration inputs | `localization_inputs` |
| `localize` | typed localization inputs | a localization run under `runs/` |

Viewer generation and index refresh are derived presentation work, not
scientific stages or completion markers.

## Artifact contract

`artifact.py` owns `farfield.artifact.v1`. A complete artifact declares:

- its kind, dataset, version, schema, generator, and producing commit;
- an exact output inventory and digest for every byte;
- its resolved scientific config and build identity;
- exact typed upstream `ArtifactRef`s, including manifest/content digests.

Publication occurs in a no-clobber `.incomplete` sibling, validates the staged
payload, then atomically makes the final directory visible. Readers reopen
the manifest and payload; directory existence alone never means complete.

The build identity is intentionally strict, but downstream code evolution
does not require recomputing an unaffected prefix. Use the reviewed reuse
bridge described below instead of copying artifacts or rewriting manifests.

## Stage-scoped reuse

When a successor build changes only inputs first consumed after tracking—for
example a catalog or identity-review policy—the existing extraction/tracking
prefix may be reused exactly:

```
bazel run //...farfield:pipeline -- prove-stage-reuse \
    --source_build_dir <source-build> \
    --target_build_dir <successor-build> \
    --through track \
    --prefix_code_reviewed_by <reviewer> \
    --prefix_code_reviewed_at <timestamp> \
    --prefix_code_review_note <reason>
```

The resulting `stage_reuse.json` is not an alias or a re-stamped artifact. It
binds the source and target build records, protected inputs, code-compatibility
attestation, and exact pinhole/frame/track references. It replays the original
producer validations, rejects changes consumed by the prefix, and authorizes
only the listed artifacts through `track`. Every direct downstream consumer
independently revalidates the proof and records the bridge in its own
provenance. Removing, changing, relocating, or widening the proof fails
closed.

## Provider stages and recovery

Extraction, audit, and matching use the shared lifecycle in
`llm_lifecycle.py`:

1. build an immutable, content-addressed `RequestSet`;
2. retain every provider attempt separately;
3. validate exact key coverage and response shape;
4. publish the canonical result only when every request has exactly one valid
   result.

For audit and matching, `--build_only` seals requests without provider
execution. Submission and aggregation are separate operations so a completed
transport response is not rebilled merely because canonicalization failed.
Retry work is explicit and all attempts remain auditable. Partial coverage
never masquerades as an empty or complete scientific artifact.

Transport-only fields, including supported provider thought signatures, are
validated and accounted for at the transport boundary. They do not silently
become canonical scientific fields.

Semantic audit asks the provider for one decision code and deterministically
maps it to the stable canonical audit schema. This prevents contradictory
combinations of verdict, identity, and drop reason. Controlled losses remain
allowed: invalid responses or unusable evidence are counted and reported,
while systematic rates trigger investigation instead of silent repair.

Matching publishes uncertainty-preserving machine hypotheses and a null
alternative. With no identity-review directory configured, candidates are
not promoted to human-confirmed identities. A human-assisted lane is a
separate, explicitly labeled experiment.

## Retained extraction evidence

`extraction/legacy_extraction_adoption.py` can adopt previously paid provider
responses without contacting a provider. Its default is report-only. It
requires exact enumerated request/result sources, reconstructs the current
request workload and pinhole bytes, validates primary/retry coverage and
provider echoes, and emits complete normalization and sanitation ledgers.

Explicit publication creates typed REQUEST -> canonical RESULT lineage and a
FRAME artifact with direct PINHOLE + RESULT upstreams. Prefix publication is
resumable only by reopening and exactly validating the already-published
prefix; collisions or gaps fail before any suffix write.

## Tracking and viewers

Tracking never seeds a new track from an unclaimed detection on the final
interval, because such a track could never have a record. Existing tracks are
still updated by matching final detections. Producer validation rejects every
missing, malformed, or empty track record array.

After a track artifact publishes, the launcher attempts a separate typed
`frame_landmark_viewer` sidecar bound to the exact frame and track artifacts.
A rendering failure is reported without invalidating tracking. This keeps
viewer code and HTML out of scientific hashes while preserving per-item
integrity and bidirectional frame/track navigation.
