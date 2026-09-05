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

Reuse needs no bridge and no attestation. An artifact is identified by what
determines it, per stage:

    artifact_identity = H(kind, dataset,
                          that stage's own resolved config,
                          its upstreams' manifest digests,
                          the recorded build inputs it reads)

The identity is recorded in each artifact's manifest at publish time, beside
the recipe that produced it. A build that names an existing version of an
artifact is plugging a prior part in, and the artifact is checked from its
own manifest:

- it is self-describing: the identity recomputed from its recorded recipe
  equals the identity it records;
- the settings this build states for that stage agree with the settings the
  artifact was made with. A shared setting with a different value is a
  different artifact and is refused: configure the values it was made with,
  or name a new version. A setting the artifact predates cannot have shaped
  it and is reported as unverified, not refused. `execution` and `cost` say
  how a provider was reached and what it could cost; they never gate;
- it binds the sibling artifacts this build names: an audit built from one
  tracks version cannot be plugged in beside another.

Nothing is recomputed from the consuming build's recipe. Changing a knob only
the last stage reads therefore does not move the identity of the paid
extraction or the hours of tracking upstream of it, and adding a setting to a
stage does not orphan every artifact that stage published before the setting
existed. The producing stage still refuses to publish over a version that was
made with different settings.

Code is deliberately NOT part of that identity. `code_provenance` records the
commit and diff that produced every artifact, and `pipeline status` reports
whether a build's artifacts share one code state, but nothing is ever refused
for it: code changes constantly in a research tree and data does not. See
`docs/farfield/decisions.md` for the measurements behind that.

## An artifact answers for itself

Two questions have to be answerable about anything on disk: how do I
reproduce it, and are the things it was made from out of date. Both are
answered by the artifact's own manifest, with no join to a build directory.

`artifact_recipe` records the terms of the identity a manifest could not
otherwise recover -- the stage's resolved config, the build inputs that stage
read, and which upstream digests entered the identity. `manifest.upstreams`
is the fuller lineage record and the identity term is a subset of it: a
`frame_landmarks` manifest records its pinhole artifact and the canonical LLM
result artifact, while `extract` declares no artifact upstreams at all.

```
bazel run //...farfield:pipeline -- recipe --artifact_dir <artifact>
```

prints the settings, inputs and lineage, then verifies the property that
makes the record trustworthy: **the identity recomputed from the manifest
alone equals the identity the manifest records.** A manifest missing a term
its identity depends on cannot satisfy that, which is why it is a check and
not a convention.

`builds/<dataset>/<build>/build_config.json` remains the recipe a run is
driven from, and it is still where a whole multi-stage config lives. It is no
longer load-bearing for reproducing any single artifact.

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

Some `frame_landmarks` artifacts were adopted from previously paid provider
responses rather than produced by a live extraction, and they say so: their
manifest carries the `legacy_adoption` generator and config schema, and the
digest of the verification report that was produced when they were adopted.
`extract_landmarks` recognises and validates them against that recorded
digest.

The adoption tool itself is gone. It re-verified the whole workload on every
read, which existed to satisfy a whole-build identity check that no longer
exists; the recorded report digest is what the claim actually rests on. An
artifact adopted this way is a weaker claim than one this pipeline built, and
its manifest is where that shows.

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

## Ablation producers

`tracking:detections_as_tracks --build_dir <build>` stands in for the
`track` and `audit` stages: every ingested detection becomes one
single-record `object_tracks` track (its pano box as the mask box) and one
deterministic `semantic_audits` record that restates the detection's own
tags, with no provider call. The artifacts record the identities the
orchestrator computes for those stages, so `pipeline run --from bearings`
consumes them unchanged; give them a lane version dedicated to the
ablation. Matching recognises the audit manifest's
`audit_source: detection_passthrough_v1` and switches to a single-detection
Set 1 prompt with one entry per distinct tag bundle. Runs of such a build
must carry an `ablation_tags` entry; they are diagnostic controls, not
evaluations.
