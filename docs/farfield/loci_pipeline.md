# Far-field LOCI pipeline

This is the start-to-finish runbook for applying the released LOCI late-fusion
pipeline to the far-field datasets. It covers the scientific inputs, reusable
artifacts, inference branches, path protocol, and histogram-filter outputs.

The input builders and their imagery provenance are documented in the
[far-field LOCI input guide](../../experimental/overhead_matching/swag/farfield/loci/README.md).
The separate [far-field pipeline](pipeline.md) describes the bearing-based
tracking and localization system; it is not the LOCI pipeline described here.

## Completed baseline scope

The released, zero-shot LOCI model has been run over seven recorded legs and
an approximately 150 km² search region for each geographic scope:

- Charles River: one leg and one map scope.
- Boston Harbor: three legs sharing one map scope.
- Mount Washington: three legs sharing one map scope.

No LOCI model training was performed. After the prepared VLM inputs, the only
paid external work was embedding tag values absent from the released
correspondence dictionary: 11,687 values in 47 direct Vertex embedding
batches. That operation used `embed_content` and required no upload to GCS.

## Pipeline and reuse boundaries

```text
                              shared across all seven legs
released 768-D dictionary + OSM tags + VLM tag strings
                              |
                              v
                    extended 768-D dictionary
                              |
             +----------------+----------------+
             |                                 |
             | per leg                         | per geographic scope
             v                                 v
VLM panorama tags + shared OSM       satellite images + released WAG
             |                                 |
             v                                 v
correspondence P(match) scores          satellite WAG embeddings
             |                                 |
             v                                 | reused by every leg in scope
semantic pano x satellite matrix               |
             |                                 v
             |                    leg panoramas -> visual pano x satellite matrix
             |                                 |
             +----------------+----------------+
                              |
                              v
              LOCI late-fusion observation likelihood
                              |
              full-leg forward/reverse trajectory
                              |
                              v
                   histogram Bayes filter
                              |
                              v
            beliefs, errors, convergence, provenance
```

The reuse boundary is intentional:

- Region, satellite, and OSM artifacts are shared by the Boston legs and by
  the Mount Washington legs.
- The 768-D tag-value dictionary is shared by every leg.
- Satellite WAG embeddings are computed once per map scope. Panorama WAG
  embeddings and both final similarity matrices remain leg-specific.
- VLM annotations, raw correspondence scores, paths, and filter results remain
  leg-specific because their panorama sequences differ.

Do not run two jobs that may create the same shared satellite-embedding file at
the same time. Create it with one leg, validate it, then let the other legs read
it.

## Prepared input inventory

All input versions in this table are immutable. Do not substitute similarly
named retired versions.

| map scope | area | satellite artifact | patches | OSM artifact | OSM landmarks |
|---|---:|---|---:|---|---:|
| `charles_river_20260727` | 150.00 km² | `area150km2_massgis2025_z20_nearestpx_v2` | 120,744 | `area150km2_osm260101_v2` | 242,420 |
| `boston_harbor_shared` | 153.77 km², containment-limited | `area150km2_usgsnaip2023_z20_nearestpx_v2` | 123,794 | `area150km2_osm260101_v1` | 56,505 |
| `mount_washington_20260815_shared` | 150.00 km² | `area150km2_usgsnaip2023_z20_nearestpx_v2` | 128,522 | `area150km2_z20_osm260101_v2` | 2,970 |

Every active satellite artifact contains 640 x 640 patches with a 640-source-pixel
footprint at z20. Charles uses MassGIS 2025 imagery. Boston Harbor and Mount
Washington use pinned 2023 USGS NAIP rasters with complete coverage of their
rendered footprints.

Every leg uses VLM version
`panov2_tuned_prompt_gemini3flash_2048_ultrahigh_v1`:

| leg | panoramas | VLM landmarks |
|---|---:|---:|
| `charles_river_20260727` | 513 | 2,225 |
| `boston_harbor_leg1` | 379 | 798 |
| `boston_harbor_leg2` | 236 | 269 |
| `boston_harbor_leg3` | 734 | 1,616 |
| `mount_washington_20260815_leg1` | 134 | 232 |
| `mount_washington_20260815_leg2` | 265 | 856 |
| `mount_washington_20260815_leg3` | 398 | 1,260 |

The common root is `/data/farfield_matching`; the three exact map payloads are:

```text
artifacts/loci_satellite/charles_river_20260727/area150km2_massgis2025_z20_nearestpx_v2
artifacts/loci_osm_landmarks/charles_river_20260727/area150km2_osm260101_v2

artifacts/loci_satellite/boston_harbor_shared/area150km2_usgsnaip2023_z20_nearestpx_v2
artifacts/loci_osm_landmarks/boston_harbor_shared/area150km2_osm260101_v1

artifacts/loci_satellite/mount_washington_20260815_shared/area150km2_usgsnaip2023_z20_nearestpx_v2
artifacts/loci_osm_landmarks/mount_washington_20260815_shared/area150km2_z20_osm260101_v2
```

### Input production

For a new geographic scope, produce these inputs in order before entering
Stage 0 below. The [input guide](../../experimental/overhead_matching/swag/farfield/loci/README.md)
contains the exact commands, provider pins, recovery rules, and validation
contracts used for the current datasets.

| producer | inputs | output |
|---|---|---|
| `farfield/loci:region` | complete catalog plus every trajectory that will share the map | trajectory-containing metric inset and exact Web-Mercator lattice |
| `farfield/loci:osm` | region and complete OSM catalog | LOCI-pruned `landmarks.feather` intersecting the full patch footprint |
| `farfield/loci:satellite` | region and pinned imagery source | complete z20 patch set, source/patch hashes, and coverage evidence |
| `semantic_landmark_extractor create_panorama_sentences` plus `farfield/loci:vlm_requests` | adopted 2,048 px pinholes and stock LOCI prompt | one validated four-image request per panorama |
| Vertex batch lifecycle | immutable requests and `gemini-3-flash-preview` | complete canonical structured OSM-tag responses |
| VLM embedding builder | canonical responses | one validated 1,536-D pano-v2 pickle per leg |

The far-field faces are camera-relative because these datasets do not contain
an authoritative per-frame world camera heading. That is acceptable for the
tag-only LOCI correspondence branch used here, which does not consume yaw. Do
not reinterpret it into a compass-bearing annotation contract.

## The two embedding contracts

There are two unrelated embedding types in this pipeline. Their dimensions are
not interchangeable.

| embedding | model and dimension | owner | how LOCI uses it |
|---|---|---|---|
| VLM annotation embedding | `gemini-embedding-001`, 1,536-D, `SEMANTIC_SIMILARITY` | one `loci_vlm_annotations/<leg>/.../embeddings/embeddings.pkl` per leg | Preserves the released pano-v2 annotation contract and side features. The late-fusion correspondence export reads the structured `panoramas` tag bundles from this pickle; it does not feed these 1,536-D arrays to the classifier. |
| Correspondence tag-value embedding | `text-embedding-005`, 768-D, `SEMANTIC_SIMILARITY` | one shared `loci_text_value_embeddings` artifact | The released correspondence classifier embeds each recognized OSM/VLM tag value through this table. Every value used at inference must be present. |

Never use `--allow_missing_text_embeddings` for a scientific run. It silently
replaces missing values with zero vectors and changes the model input.

## Canonical output layout

Use these immutable output lanes and version names:

```text
artifacts/
  loci_text_value_embeddings/
    charles_boston_washington/text_embedding_005_768_v1/
      text_value_embeddings.pkl
      manifest.json

  loci_correspondence_scores/
    <leg>/simple_v1_v5_text005_768_hungarian08_v1/
      raw.pt
      raw_cost_matrix.npy
      hungarian08_similarity.pt
      hungarian08_similarity.json
      manifest.json

  loci_wag_similarity/
    <map-scope>/paper_wag_no_hinge_v1/
      satellite_embeddings.pt
    <leg>/paper_wag_no_hinge_v1/
      similarity.pt
      similarity.json
      manifest.json

  loci_eval_paths/
    <leg>/full_leg_forward_reverse_v1/
      paths.json
      manifest.json

  loci_runs/
    <leg>/paper_sigmas_full_leg_v1/
      aggregator_config.yaml
      args.json
      histogram_config.json
      summary_statistics.json
      ...per-path outputs...
      manifest.json
    <leg>/paper_sigmas_full_leg_mass100_500_v3/
      0000000/metrics.json  # forward
      0000001/metrics.json  # backward
      ...mapping-truth replay outputs and manifest...
```

The JSON matrix-identity sidecars shown above are part of the current writer
contract. The sealed baseline v1 correspondence matrices and the existing
visual `similarity.json` metadata predate that identity field; their explicit
legacy handling is documented in Stage 5.

For Charles, `<map-scope>` and `<leg>` are the same directory, so the shared
satellite embeddings and leg-specific similarity matrix coexist in that
version directory. A Boston or Washington leg manifest must reference the
shared scope's satellite-embedding digest rather than copying the file.

All commands below use the same publication contract. `*_DEST` names the
immutable final directory and the corresponding `*_STAGE` is its
`<version>.incomplete` sibling. A producer writes only beneath `*_STAGE`;
after a content and alignment audit, `publish_staged` is given `*_DEST` and
atomically renames the staging sibling. It refuses to overwrite either an
existing final artifact or an existing manifest. The `v1` names below identify
the completed baseline; to reproduce it on the same artifact root, choose a
new version rather than removing or writing through the published directory.

The `--config-json` examples record the result-shaping settings needed to make
the publication calls runnable. Before publication, augment that object with
the audited counts, shapes, hashes, and ordered-ID digests listed in
[Required provenance and validation](#required-provenance-and-validation).
`publish_staged` checks the exact file list and typed upstream manifests, but it
does not replace the tensor/content audit.

## Checkpoints and frozen evaluation settings

Use the existing released/local weights; do not retrain.

| role | path | identity |
|---|---|---|
| correspondence classifier | `/data/overhead_matching/hf_release_staging/checkpoints/correspondence_classifier/best_model.pt` | SHA-256 `1d45cb6b04f2edfd847160f90c70ad9b2249bb7a192401eb815f9179833d84c3`; byte-identical to local `simple_v1_v5/best_model.pt` |
| WAG, full training output | `/data/overhead_matching/training_outputs/260215_baseline_retraining/260421_221726_all_chicago_dinov3_wag_bs18_v2_no_hinge` | Matches the released paper WAG config except for output-directory paths; `best_panorama/model.pt` SHA-256 `bc24f9401797617d90d6fe3c4fe8bb58673dc8c59dac1f337e166e3a5e24e880`, `best_satellite/model.pt` SHA-256 `ad760c668015faf4a4884d30c614276425c340557fdd78f753f4e74355baa3a2` |
| base tag dictionary | `/data/overhead_matching/hf_release_staging/correspondence/text_value_embeddings.pkl` | Released 206,277-entry `text-embedding-005` table; SHA-256 `e42b103296390b3724fab8dc3410ca345ef550a7a46289baea5e3a0b991e580d` |

Use the paper-frozen late-fusion and filter settings for the first zero-shot
baseline:

```yaml
kind: SafaPlusNormalizedLandmarkAggregatorConfig
image_similarity_matrix_path: <leg WAG similarity.pt>
landmark_similarity_matrix_path: <leg hungarian08_similarity.pt>
image_sigma: 0.1809
landmark_sigma: 0.4673
landmark_use_raw_residual: false
```

The filter settings are seed 42, motion noise 0.141 m/sqrt(m), odometry noise
0.141 m/sqrt(m) with seed 7919, subdivision factor 4, convergence radii
25/50/100 m, and a 2 GiB cell-to-patch mapping chunk limit. The prior is uniform
over the full rendered map support. The `initial_std_deg` field written by the
legacy evaluator is not the prior; `HistogramBelief.from_uniform` owns
initialization.

Do not calibrate either sigma on these target legs for the baseline. A later
target-calibrated experiment must use held-out data and a different artifact
version so it is not reported as zero-shot.

## Stage 0: validate the prepared inputs

Run `input_smoke` once per leg with the appropriate shared map artifacts. It
opens and hashes the region, satellite, and OSM artifacts, constructs the real
`VigorDataset`, checks the z20 lattice and expected counts, and requires every
panorama to have a satellite association.

```bash
FF_ROOT=/data/farfield_matching
LEG=charles_river_20260727
SAT_ARTIFACT="$FF_ROOT/artifacts/loci_satellite/charles_river_20260727/area150km2_massgis2025_z20_nearestpx_v2"
OSM_ARTIFACT="$FF_ROOT/artifacts/loci_osm_landmarks/charles_river_20260727/area150km2_osm260101_v2"

bazel run //experimental/overhead_matching/swag/farfield/loci:input_smoke -- \
  --dataset_dir "$FF_ROOT/datasets/$LEG" \
  --satellite_artifact "$SAT_ARTIFACT" \
  --osm_artifact "$OSM_ARTIFACT"
```

Repeat with the Boston shared payloads for each Boston leg and the Washington
shared payloads for each Washington leg. Also require before inference:

- each VLM `execution_manifest.json` and `embeddings/manifest.json` says
  `complete: true` and `validation.status: PASS`;
- its panorama keys exactly equal the ordered dataset panorama-ID set;
- its model is `gemini-embedding-001`, task is `SEMANTIC_SIMILARITY`, and
  output dimension is 1,536;
- the semantic and visual matrices for a leg bind the exact same ordered panorama IDs and
  satellite patch IDs.

`input_smoke` currently covers the map/dataset half of this gate. The
downstream artifact validators must cover the VLM and ordered-ID checks.

## Stage 1: build the shared 768-D tag-value dictionary

Extend the release dictionary with the union of recognized tag values from all
three OSM artifacts and all seven VLM artifacts. The collector accepts direct,
versioned VLM roots; repeated panorama IDs such as `f0000` across independent
legs are not an error because this stage unions tag strings rather than joining
panoramas. The audited collection contains 2,250 non-empty panorama tag records
from 2,659 complete panorama payloads, and 38,135 relevant unique values. Of
those values, 26,448 were already in the release dictionary and 11,687 required
a new Vertex embedding. The published output retains all 206,277 base entries,
adds those 11,687 entries, and contains 217,964 finite float32 vectors. Its
SHA-256 is `78ac11db800f9c3281a27cd6de52b0fe60c277e743b570241224c96cff31c36b`.

```bash
FF_ROOT=/data/farfield_matching
BASE_TEXT=/data/overhead_matching/hf_release_staging/correspondence/text_value_embeddings.pkl
TEXT_DEST="$FF_ROOT/artifacts/loci_text_value_embeddings/charles_boston_washington/text_embedding_005_768_v1"
TEXT_STAGE="$TEXT_DEST.incomplete"
TEXT_OUT="$TEXT_STAGE/text_value_embeddings.pkl"
VLM_VERSION=panov2_tuned_prompt_gemini3flash_2048_ultrahigh_v1
CHARLES_OSM="$FF_ROOT/artifacts/loci_osm_landmarks/charles_river_20260727/area150km2_osm260101_v2"
BOSTON_OSM="$FF_ROOT/artifacts/loci_osm_landmarks/boston_harbor_shared/area150km2_osm260101_v1"
WASHINGTON_OSM="$FF_ROOT/artifacts/loci_osm_landmarks/mount_washington_20260815_shared/area150km2_z20_osm260101_v2"

if [[ -e "$TEXT_DEST" || -e "$TEXT_STAGE" ]]; then
  echo "choose a fresh dictionary version or deliberately resume its staging directory" >&2
  exit 1
fi

TEXT_CMD=(bazel run //experimental/overhead_matching/swag/scripts:precompute_value_embeddings -- \
  --feather_dirs \
    "$CHARLES_OSM" \
    "$BOSTON_OSM" \
    "$WASHINGTON_OSM" \
  --pano_v2_base \
    "$FF_ROOT/artifacts/loci_vlm_annotations/charles_river_20260727/$VLM_VERSION" \
    "$FF_ROOT/artifacts/loci_vlm_annotations/boston_harbor_leg1/$VLM_VERSION" \
    "$FF_ROOT/artifacts/loci_vlm_annotations/boston_harbor_leg2/$VLM_VERSION" \
    "$FF_ROOT/artifacts/loci_vlm_annotations/boston_harbor_leg3/$VLM_VERSION" \
    "$FF_ROOT/artifacts/loci_vlm_annotations/mount_washington_20260815_leg1/$VLM_VERSION" \
    "$FF_ROOT/artifacts/loci_vlm_annotations/mount_washington_20260815_leg2/$VLM_VERSION" \
    "$FF_ROOT/artifacts/loci_vlm_annotations/mount_washington_20260815_leg3/$VLM_VERSION" \
  --base_embeddings "$BASE_TEXT" \
  --model text-embedding-005 \
  --output_dimensionality 768 \
  --output "$TEXT_OUT")
"${TEXT_CMD[@]}"
TEXT_PRODUCER_COMMAND="$(printf '%q ' "${TEXT_CMD[@]}")"
```

The paid portion uses Vertex `embed_content` directly; it does not require a
GCS batch upload. The runtime needs application-default credentials plus
`GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION=global`. The client is
constructed with `vertexai=True`, so `GOOGLE_GENAI_USE_VERTEXAI` is not
required.

Publish only after re-scanning all ten sources against the output and proving
zero missing values, 768 finite float32 components per entry, preservation of
every base key/value, and an exact output hash. The manifest must also record
the model, task `SEMANTIC_SIMILARITY`, dimensions, base dictionary hash, all ten
source artifact hashes, counts, code provenance, and Vertex project/location.
The three OSM inputs are typed artifact upstreams; the seven VLM roots and the
released base pickle are not, so bind their exact manifest/file hashes in the
audited config JSON.

```bash
TEXT_CONFIG_JSON='{"model":"text-embedding-005","task_type":"SEMANTIC_SIMILARITY","output_dimensionality":768,"base_embeddings_sha256":"e42b103296390b3724fab8dc3410ca345ef550a7a46289baea5e3a0b991e580d","validation":{"all_values_present":true,"base_preserved_exactly":true,"dtype":"float32","finite":true}}'

bazel run //experimental/overhead_matching/swag/farfield/loci:publish_staged -- \
  --destination "$TEXT_DEST" \
  --generator //experimental/overhead_matching/swag/scripts:precompute_value_embeddings \
  --producer-command "$TEXT_PRODUCER_COMMAND" \
  --declared-output text_value_embeddings.pkl \
  --upstream "$CHARLES_OSM" \
  --upstream "$BOSTON_OSM" \
  --upstream "$WASHINGTON_OSM" \
  --config-json "$TEXT_CONFIG_JSON"
```

## Stage 2: export semantic correspondence similarity

This branch uses the released classifier, the shared 768-D dictionary, one
leg's VLM tags, and the scope's shared OSM geometry. It has two products:

1. Raw `P(match)` for every panorama-landmark x OSM-landmark pair.
2. A panorama x satellite-patch matrix made by Hungarian matching, summation,
   a 0.8 probability threshold, and panorama-landmark uniqueness weighting.

Use a streamed NumPy matrix for every leg. It is essential for Charles: its
2,225 x 242,420 raw matrix contains about 539 million float32 scores, or
approximately 2.01 GiB (2.16 GB), before metadata and the aggregated matrix.

```bash
FF_ROOT=/data/farfield_matching
LEG=charles_river_20260727
VLM_VERSION=panov2_tuned_prompt_gemini3flash_2048_ultrahigh_v1
VLM_ROOT="$FF_ROOT/artifacts/loci_vlm_annotations/$LEG/$VLM_VERSION"
SAT_ARTIFACT="$FF_ROOT/artifacts/loci_satellite/charles_river_20260727/area150km2_massgis2025_z20_nearestpx_v2"
OSM_ARTIFACT="$FF_ROOT/artifacts/loci_osm_landmarks/charles_river_20260727/area150km2_osm260101_v2"
TEXT_EMBEDDINGS="$FF_ROOT/artifacts/loci_text_value_embeddings/charles_boston_washington/text_embedding_005_768_v1/text_value_embeddings.pkl"
TEXT_ARTIFACT="${TEXT_EMBEDDINGS%/text_value_embeddings.pkl}"
CORR_MODEL=/data/overhead_matching/hf_release_staging/checkpoints/correspondence_classifier/best_model.pt
CORR_DEST="$FF_ROOT/artifacts/loci_correspondence_scores/$LEG/simple_v1_v5_text005_768_hungarian08_v1"
CORR_STAGE="$CORR_DEST.incomplete"

if [[ -e "$CORR_DEST" || -e "$CORR_STAGE" ]]; then
  echo "choose a fresh correspondence version or deliberately resume its staging directory" >&2
  exit 1
fi

CORR_RAW_CMD=(bazel run //experimental/overhead_matching/swag/scripts:export_correspondence_similarity -- \
  --model_path "$CORR_MODEL" \
  --text_embeddings_path "$TEXT_EMBEDDINGS" \
  --dataset_path "$FF_ROOT/datasets/$LEG" \
  --pano_v2_base "$VLM_ROOT" \
  --satellite_dir "$SAT_ARTIFACT/satellite" \
  --landmark_path "$OSM_ARTIFACT/landmarks.feather" \
  --output_path "$CORR_STAGE/raw.pt" \
  --stream_cost_matrix)
"${CORR_RAW_CMD[@]}"

CORR_AGG_CMD=(bazel run //experimental/overhead_matching/swag/scripts:export_correspondence_similarity -- \
  --from_raw "$CORR_STAGE/raw.pt" \
  --dataset_path "$FF_ROOT/datasets/$LEG" \
  --satellite_dir "$SAT_ARTIFACT/satellite" \
  --landmark_path "$OSM_ARTIFACT/landmarks.feather" \
  --output_path "$CORR_STAGE/hungarian08.pt" \
  --compute_similarity \
  --method hungarian \
  --aggregation sum \
  --prob_threshold 0.8 \
  --uniqueness_weighted)
"${CORR_AGG_CMD[@]}"

CORR_PRODUCER_COMMAND="$(printf '%q ' "${CORR_RAW_CMD[@]}"); $(printf '%q ' "${CORR_AGG_CMD[@]}")"
```

The second command writes `hungarian08_similarity.pt` and
`hungarian08_similarity.json`; the sidecar binds the tensor's exact ordered
panorama and satellite identities plus the aggregation policy. Keeping the raw
output allows a matching-policy change to be evaluated without rerunning
classifier inference.

Publishing renames `<version>.incomplete` to its final directory, so streamed
raw metadata must not depend on an absolute staging path. The current writer
records `raw_cost_matrix.npy` relative to `raw.pt`; the loader also supports
older metadata by falling back to that sibling file when a recorded absolute
path is stale. The seven published v1 artifacts use that tested fallback.
Keep the `.pt` and `.npy` files together when moving an artifact.

New `raw.pt` metadata also carries
`swag_raw_correspondence_identity/v1`. It binds the exact trajectory mapping,
ordered panorama and satellite IDs, satellite-to-landmark associations, OSM
source bytes, raw tag/index metadata and score values, classifier, text
dictionary, and ordered VLM pickle hashes. The score-value digest is streamed
in bounded chunks, including for the Charles memmap. The aggregation command
verifies the live dataset/OSM/map portion and raw payload before it is allowed
to stamp a matrix identity. This prevents a raw file from another leg or a
same-shaped replaced score matrix from being relabeled as plausible output.
The sealed v1 raw artifacts predate this contract; only when deliberately
re-aggregating one of those already audited files, add
`--allow_legacy_raw_identity` to `CORR_AGG_CMD`. Never add that flag for a new
raw artifact, and a present identity mismatch is always fatal.

The artifact is complete only when the raw dimensions, aggregated dimensions,
finite-value contract, and ordered panorama/satellite IDs validate. The
histogram evaluator recomputes the ordered-ID digests from the live dataset and
rejects a mismatch even when the tensor shape is unchanged. A patch with no
usable landmark evidence receives zero; an all-zero or constant semantic row
makes the normalized LOCI aggregator use image-only evidence for that
panorama. Unexpected missing tag embeddings are fatal.

After that audit, publish the four files together. `TEXT_ARTIFACT`,
`SAT_ARTIFACT`, and `OSM_ARTIFACT` are completed typed artifacts; the dataset,
VLM payload, and checkpoint are bound by exact hashes in the augmented config.

```bash
CORR_CONFIG_JSON='{"classifier_sha256":"1d45cb6b04f2edfd847160f90c70ad9b2249bb7a192401eb815f9179833d84c3","stream_cost_matrix":true,"aggregation":{"method":"hungarian","reduction":"sum","probability_threshold":0.8,"uniqueness_weighted":true,"dustbin":true},"validation":{"finite":true,"ordered_ids_match":true}}'

bazel run //experimental/overhead_matching/swag/farfield/loci:publish_staged -- \
  --destination "$CORR_DEST" \
  --generator //experimental/overhead_matching/swag/scripts:export_correspondence_similarity \
  --producer-command "$CORR_PRODUCER_COMMAND" \
  --declared-output raw.pt \
  --declared-output raw_cost_matrix.npy \
  --declared-output hungarian08_similarity.pt \
  --declared-output hungarian08_similarity.json \
  --upstream "$TEXT_ARTIFACT" \
  --upstream "$SAT_ARTIFACT" \
  --upstream "$OSM_ARTIFACT" \
  --config-json "$CORR_CONFIG_JSON"
```

## Stage 3: export WAG visual similarity

Run the exact paper WAG checkpoint over the equirectangular leg panoramas and
the scope's 640 x 640 satellite patches. `--disable_safa_cache` is required:
the legacy tensor-cache namespace does not bind the external satellite
artifact identity. The separate reusable satellite cache does bind that
identity: new caches use schema `swag_satellite_embeddings/v2` and cover the
ordered filenames and image bytes, satellite weights and model configuration,
tag-embedding override, preprocessing, and satellite forward-code content.
Cache writes are atomic, so an interrupted write is never mistaken for a
complete cache.

```bash
FF_ROOT=/data/farfield_matching
LEG=mount_washington_20260815_leg1
MAP_SCOPE=mount_washington_20260815_shared
SAT_ARTIFACT="$FF_ROOT/artifacts/loci_satellite/$MAP_SCOPE/area150km2_usgsnaip2023_z20_nearestpx_v2"
WAG_MODEL=/data/overhead_matching/training_outputs/260215_baseline_retraining/260421_221726_all_chicago_dinov3_wag_bs18_v2_no_hinge
WAG_SHARED_DEST="$FF_ROOT/artifacts/loci_wag_similarity/$MAP_SCOPE/paper_wag_no_hinge_v1"
WAG_SHARED_STAGE="$WAG_SHARED_DEST.incomplete"
WAG_DEST="$FF_ROOT/artifacts/loci_wag_similarity/$LEG/paper_wag_no_hinge_v1"
WAG_STAGE="$WAG_DEST.incomplete"

for path in "$WAG_SHARED_DEST" "$WAG_SHARED_STAGE" "$WAG_DEST" "$WAG_STAGE"; do
  if [[ -e "$path" ]]; then
    echo "choose a fresh WAG version or deliberately resume its staging directory: $path" >&2
    exit 1
  fi
done

WAG_CMD=(bazel run //experimental/overhead_matching/swag/scripts:export_similarity_matrix -- \
  --model_path "$WAG_MODEL" \
  --checkpoint best \
  --dataset_path "$FF_ROOT/datasets/$LEG" \
  --satellite_dir "$SAT_ARTIFACT/satellite" \
  --satellite_embeddings_path "$WAG_SHARED_STAGE/satellite_embeddings.pt" \
  --output_path "$WAG_STAGE/similarity.pt" \
  --disable_safa_cache)
"${WAG_CMD[@]}"
WAG_PRODUCER_COMMAND="$(printf '%q ' "${WAG_CMD[@]}")"
```

`SAT_ARTIFACT` is not derivable from `MAP_SCOPE` alone because Charles uses a
different provider/version name from Boston and Washington. Select the exact
scope-specific path from the prepared-input table above; never carry the
Charles MassGIS suffix into a NAIP scope.

For the first Boston or Washington leg, that one exporter invocation creates
two distinct staging artifacts: the scope cache and the leg matrix. Audit both,
publish the scope cache first, and only then publish the leg matrix with the
completed scope cache as an upstream. The exporter records the future published
cache path in `similarity.json`, even though the cache was built in its staging
sibling.

```bash
WAG_SHARED_CONFIG_JSON='{"checkpoint":"best_satellite","disable_safa_cache":true,"schema":"swag_satellite_embeddings/v2","validation":{"finite":true,"ordered_satellite_files_match":true}}'

bazel run //experimental/overhead_matching/swag/farfield/loci:publish_staged -- \
  --destination "$WAG_SHARED_DEST" \
  --generator //experimental/overhead_matching/swag/scripts:export_similarity_matrix \
  --producer-command "$WAG_PRODUCER_COMMAND" \
  --declared-output satellite_embeddings.pt \
  --upstream "$SAT_ARTIFACT" \
  --config-json "$WAG_SHARED_CONFIG_JSON"

WAG_LEG_CONFIG_JSON='{"checkpoint":"best","disable_safa_cache":true,"satellite_embeddings_source":"computed","identity_sidecar":"similarity.json","validation":{"finite":true,"ordered_ids_match":true}}'

bazel run //experimental/overhead_matching/swag/farfield/loci:publish_staged -- \
  --destination "$WAG_DEST" \
  --generator //experimental/overhead_matching/swag/scripts:export_similarity_matrix \
  --producer-command "$WAG_PRODUCER_COMMAND" \
  --declared-output similarity.pt \
  --declared-output similarity.json \
  --upstream "$WAG_SHARED_DEST" \
  --upstream "$SAT_ARTIFACT" \
  --config-json "$WAG_LEG_CONFIG_JSON"
```

For a later leg in that scope, require the completed cache and point the
exporter at it. Only a new leg staging directory is created; publish it with
the same leg call above after changing `satellite_embeddings_source` to
`reused` and capturing the new `WAG_PRODUCER_COMMAND`.

```bash
LEG=mount_washington_20260815_leg2
WAG_DEST="$FF_ROOT/artifacts/loci_wag_similarity/$LEG/paper_wag_no_hinge_v1"
WAG_STAGE="$WAG_DEST.incomplete"
test -f "$WAG_SHARED_DEST/manifest.json"
if [[ -e "$WAG_DEST" || -e "$WAG_STAGE" ]]; then
  echo "choose a fresh leg WAG version or deliberately resume its staging directory" >&2
  exit 1
fi

WAG_CMD=(bazel run //experimental/overhead_matching/swag/scripts:export_similarity_matrix -- \
  --model_path "$WAG_MODEL" \
  --checkpoint best \
  --dataset_path "$FF_ROOT/datasets/$LEG" \
  --satellite_dir "$SAT_ARTIFACT/satellite" \
  --satellite_embeddings_path "$WAG_SHARED_DEST/satellite_embeddings.pt" \
  --output_path "$WAG_STAGE/similarity.pt" \
  --allow_legacy_satellite_embeddings \
  --disable_safa_cache)
"${WAG_CMD[@]}"
WAG_PRODUCER_COMMAND="$(printf '%q ' "${WAG_CMD[@]}")"

# Audit similarity.pt and similarity.json before this publication step.
WAG_LEG_CONFIG_JSON='{"checkpoint":"best","disable_safa_cache":true,"satellite_embeddings_source":"reused","identity_sidecar":"similarity.json","validation":{"finite":true,"ordered_ids_match":true}}'
bazel run //experimental/overhead_matching/swag/farfield/loci:publish_staged -- \
  --destination "$WAG_DEST" \
  --generator //experimental/overhead_matching/swag/scripts:export_similarity_matrix \
  --producer-command "$WAG_PRODUCER_COMMAND" \
  --declared-output similarity.pt \
  --declared-output similarity.json \
  --upstream "$WAG_SHARED_DEST" \
  --upstream "$SAT_ARTIFACT" \
  --config-json "$WAG_LEG_CONFIG_JSON"
```

The `--allow_legacy_satellite_embeddings` flag in that exact example is needed
only because `paper_wag_no_hinge_v1` is the already-published baseline cache,
whose v1 schema predates the complete behavior identity. It still validates
the ordered imagery bytes and checkpoint weights and emits a warning. Omit the
flag for every v2 cache; an identity mismatch is always fatal. A corrupt or
partial cache must be deleted and rebuilt rather than accepted as legacy.

Charles is the one-directory case because its map scope and leg have the same
name. Point both outputs into one staging directory, audit all three files, and
publish once with `satellite_embeddings.pt`, `similarity.pt`, and
`similarity.json` declared. Do not run the two-artifact first-build recipe with
identical `WAG_SHARED_DEST` and `WAG_DEST` values.

```bash
LEG=charles_river_20260727
SAT_ARTIFACT="$FF_ROOT/artifacts/loci_satellite/$LEG/area150km2_massgis2025_z20_nearestpx_v2"
WAG_DEST="$FF_ROOT/artifacts/loci_wag_similarity/$LEG/paper_wag_no_hinge_v1"
WAG_STAGE="$WAG_DEST.incomplete"
if [[ -e "$WAG_DEST" || -e "$WAG_STAGE" ]]; then
  echo "choose a fresh Charles WAG version or deliberately resume its staging directory" >&2
  exit 1
fi

WAG_CMD=(bazel run //experimental/overhead_matching/swag/scripts:export_similarity_matrix -- \
  --model_path "$WAG_MODEL" \
  --checkpoint best \
  --dataset_path "$FF_ROOT/datasets/$LEG" \
  --satellite_dir "$SAT_ARTIFACT/satellite" \
  --satellite_embeddings_path "$WAG_STAGE/satellite_embeddings.pt" \
  --output_path "$WAG_STAGE/similarity.pt" \
  --disable_safa_cache)
"${WAG_CMD[@]}"
WAG_PRODUCER_COMMAND="$(printf '%q ' "${WAG_CMD[@]}")"

# Audit the shared cache and leg matrix before this single publication step.
WAG_COMBINED_CONFIG_JSON='{"checkpoint":"best","disable_safa_cache":true,"satellite_embeddings_source":"computed","schema":"swag_satellite_embeddings/v2","identity_sidecar":"similarity.json","validation":{"finite":true,"ordered_ids_match":true}}'
bazel run //experimental/overhead_matching/swag/farfield/loci:publish_staged -- \
  --destination "$WAG_DEST" \
  --generator //experimental/overhead_matching/swag/scripts:export_similarity_matrix \
  --producer-command "$WAG_PRODUCER_COMMAND" \
  --declared-output satellite_embeddings.pt \
  --declared-output similarity.pt \
  --declared-output similarity.json \
  --upstream "$SAT_ARTIFACT" \
  --config-json "$WAG_COMBINED_CONFIG_JSON"
```

The panorama embeddings and final dot-product matrix are recomputed per leg.
Every final matrix must have shape `(leg panorama count, scope patch count)` and
the same ordered IDs as the semantic matrix. `similarity.json` carries the same
ordered matrix-identity contract as the correspondence sidecar, and binds the
ordered panorama names/bytes and panorama checkpoint state/config.
Evaluation checks the ordered matrix identity. Because `similarity.json` is a
declared output, the published artifact's content digest also seals these
input identities. Never run two first-build jobs for the same scope cache
concurrently.

## Stage 4: create full-leg evaluation paths

The far-field datasets are sequential recordings. For the first baseline,
write exactly two paths per leg from `pano_id_mapping.csv`: the complete
recorded order and its exact reverse.

```bash
FF_ROOT=/data/farfield_matching
LEG=mount_washington_20260815_leg1
PATH_DEST="$FF_ROOT/artifacts/loci_eval_paths/$LEG/full_leg_forward_reverse_v1"
PATH_OUT="$PATH_DEST.incomplete/paths.json"

bazel run //experimental/overhead_matching/swag/scripts:create_eval_paths_from_panorama_trajectory -- \
  --dataset_path "$FF_ROOT/datasets/$LEG" \
  --full_trajectory \
  --out "$PATH_OUT"
```

Validate the staging payload and publish `PATH_DEST` with
`//experimental/overhead_matching/swag/farfield/loci:publish_staged`; do not
write `paths.json` directly into the completed artifact directory.

Do not use the paper's 3 km sliding-window protocol for Mount Washington. Its
three legs are only approximately 0.39, 0.79, and 1.18 km long. Padding,
looping, or duplicating these tracks would manufacture distance and
independence. Report each direction and leg, then a clearly labeled macro
aggregate. Use the same full-leg forward/reverse construction for Charles and
Boston so all seven runs share one path contract; comparisons to the paper's
3 km/5 km path statistics remain descriptive rather than like-for-like.

The path manifest binds the ordered `pano_id_mapping.csv` hash, panorama count,
forward and reverse arrays, accumulated distance, and generator/code identity.

The published v1 artifacts have these source-bound measurements:

| leg | panoramas per direction | trajectory distance | mapping SHA-256 |
|---|---:|---:|---|
| `charles_river_20260727` | 513 | 5,468.238 m | `4b36fdabe2699e903eae9a461ec1e9e1646b695b172fadbf6c82614df0cb9d26` |
| `boston_harbor_leg1` | 379 | 13,301.491 m | `31f61235ac3f606b82f10cde980b3498b4394a6b58266caf8ab8327751391387` |
| `boston_harbor_leg2` | 236 | 6,426.712 m | `29f4b88cfcc3de5db2d16205f075dce9426145afe81d5eacc46d30f3ffcc1fb9` |
| `boston_harbor_leg3` | 734 | 18,234.172 m | `601aad01f2e71d862db263eb98316a39d3ff95a77d3a410222e1dc54c67b09af` |
| `mount_washington_20260815_leg1` | 134 | 390.159 m | `447b6cd0a9f1d0aa4925f825e20f1c757074c0b93055f08b3e5bd65301713cde` |
| `mount_washington_20260815_leg2` | 265 | 789.670 m | `26840a7a4234293fefdfe2f9358694cf15607d219941ce971a47553361cbd9ec` |
| `mount_washington_20260815_leg3` | 398 | 1,178.598 m | `7d423ab7bb26307755e164c3852c557f4512e30a884096d5eaa99f5c86b6d573` |

Every artifact passed exact forward-order, exact reverse-order, unique ID and
filename, mapped-image existence, and complete image-directory equality
checks. Distance is the sum of consecutive-row haversine distances with Earth
radius 6,378,137 m, matching the generator.

## Stage 5: fuse matrices and run the histogram filter

The seven checked-in configs under
`experimental/overhead_matching/swag/farfield/loci/configs/` freeze each leg's
matrix paths and the paper sigmas. Pass their absolute source paths: a relative
workspace path is not reliable after `bazel run` changes into its runfiles
environment. The evaluator copies the selected config into the staged result.
The external satellite directory lets it load the shared map without mutating
or symlinking the frozen dataset, and it infers `source_px` from the sibling
`satellite_bbox.json`.

Those seven configs explicitly set `allow_legacy_similarity_identity: true`
because the immutable v1 matrices were produced before ordered-identity
sidecars existed. The evaluator warns and then relies on the completed manual
alignment audit recorded below. Do not carry that option into a new run: new
exporters always write sidecars, and the default strict evaluator requires
them. If a sidecar exists, a digest mismatch remains fatal even when the legacy
option is present.

```bash
FF_ROOT=/data/farfield_matching
REPO_ROOT="$(git rev-parse --show-toplevel)"
LEG=mount_washington_20260815_leg1
SAT_ARTIFACT="$FF_ROOT/artifacts/loci_satellite/mount_washington_20260815_shared/area150km2_usgsnaip2023_z20_nearestpx_v2"
PATH_ARTIFACT="$FF_ROOT/artifacts/loci_eval_paths/$LEG/full_leg_forward_reverse_v1"
AGGREGATOR_CONFIG="$REPO_ROOT/experimental/overhead_matching/swag/farfield/loci/configs/$LEG.yaml"
RUN_DEST="$FF_ROOT/artifacts/loci_runs/$LEG/paper_sigmas_full_leg_v1"
RUN_STAGE="$RUN_DEST.incomplete"

bazel run //experimental/overhead_matching/swag/scripts:evaluate_histogram_on_paths -- \
  --aggregator-config "$AGGREGATOR_CONFIG" \
  --paths-path "$PATH_ARTIFACT/paths.json" \
  --allow-legacy-path-identity \
  --output-path "$RUN_STAGE" \
  --dataset-path "$FF_ROOT/datasets/$LEG" \
  --satellite-dir "$SAT_ARTIFACT/satellite" \
  --landmark-version v1 \
  --save-intermediate-filter-states \
  --seed 42 \
  --panorama-neighbor-radius-deg 0.0005 \
  --panorama-landmark-radius-px 640 \
  --motion-noise-frac 0.141 \
  --subdivision-factor 4 \
  --convergence-radii "25,50,100" \
  --max-chunk-gib 2.0 \
  --odometry-noise-frac 0.141 \
  --odometry-noise-seed 7919
```

Audit every staged tensor and summary, then publish `RUN_DEST` with
`//experimental/overhead_matching/swag/farfield/loci:publish_staged`. Do not
publish merely because the evaluator exited successfully. At minimum retain
`summary_statistics.json`, per-path belief/error histories, the exact
aggregator and histogram configs, argument record, and a manifest. Report
weighted-mean and MAP final error, probability mass within 25/50/100 m as a
function of distance, and the released integrated convergence-cost metrics.
A run is not complete if either matrix, a path panorama ID, or the
satellite lattice is misaligned.

The exact v1 command above needs `--allow-legacy-path-identity` because its
sealed `paths.json` files contain the historical `dataset_hash: "new"`
placeholder; their typed manifests independently bind the real mapping hashes
listed in Stage 4. Omit this flag for every newly generated path artifact. The
current evaluator hashes the live `pano_id_mapping.csv` and rejects a path file
from another leg or dataset revision even when its panorama IDs happen to be
the same.

### Completed zero-shot baseline

All seven forward/reverse runs are published and validated. Errors below are
the v3 final great-circle errors of the posterior weighted-mean and MAP-cell
estimates against full-precision `pano_id_mapping.csv` truth. The average is
over the two directions for that leg; the macro row is an unweighted average
over the seven legs.

| leg | forward mean / MAP | reverse mean / MAP | direction-average mean / MAP |
|---|---:|---:|---:|
| `charles_river_20260727` | 761.04 / 752.73 m | 430.93 / 435.81 m | 595.99 / 594.27 m |
| `boston_harbor_leg1` | 4,670.10 / 4,753.72 m | 357.93 / 269.39 m | 2,514.02 / 2,511.56 m |
| `boston_harbor_leg2` | 7,828.39 / 7,812.28 m | 7,797.25 / 7,791.61 m | 7,812.82 / 7,801.95 m |
| `boston_harbor_leg3` | 248.96 / 257.84 m | 8,697.36 / 8,536.51 m | 4,473.16 / 4,397.17 m |
| `mount_washington_20260815_leg1` | 1,782.01 / 1,780.54 m | 1,704.16 / 1,696.83 m | 1,743.08 / 1,738.69 m |
| `mount_washington_20260815_leg2` | 1,111.59 / 1,107.89 m | 1,720.32 / 1,725.46 m | 1,415.95 / 1,416.68 m |
| `mount_washington_20260815_leg3` | 60.53 / 64.94 m | 455.76 / 452.01 m | 258.15 / 258.47 m |
| **unweighted seven-leg macro** | **2,351.80 / 2,361.42 m** | **3,023.39 / 2,986.80 m** | **2,687.59 / 2,674.11 m** |

#### Far-field-compatible posterior-mass scores

The primary far-field localization metric is posterior probability mass within
500 m of truth, with 100 m as the secondary radius. Each run score is the
trapezoidal area under that mass curve over true distance travelled, divided by
trajectory length. It is a unitless number in `[0, 1]`, and higher is better.
These are two cumulative-radius metrics, not a 100--500 m annulus.

The v1 LOCI evaluations recorded only 25/50/100 m. The first v2 metric replay
inherited rounded latitude/longitude from panorama filenames and is retained
only as superseded provenance. The canonical v3 artifacts replay the same
filters with `--convergence-radii "100,500"`, use full-precision
`pano_id_mapping.csv` coordinates for evaluation truth, and then call the
existing far-field `localization.metrics.position_mass_summary` implementation
without duplicating its formula. Filter motion and odometry still use the
released LOCI filename coordinates, so this changes scoring rather than the
baseline inference algorithm.

```bash
RUN_DEST="$FF_ROOT/artifacts/loci_runs/$LEG/paper_sigmas_full_leg_mass100_500_v3"
bazel run //experimental/overhead_matching/swag/scripts:summarize_loci_position_mass -- \
  "$RUN_DEST.incomplete"
```

The replay audit found every path and LOCI mean/MAP/variance history bytewise
identical to v2. Truth-derived distance, error, and mass outputs were
deliberately recomputed. Every v3 distance matches the mapping at float64
precision; every 500 m curve is pointwise at least its 100 m curve; and all 14
stored summaries exactly match a fresh call to the canonical far-field metric.
`0000000` is forward and `0000001` is the exact backward path. The mass values
below are percentages after distance normalization; stored JSON values are
fractions.

| leg | direction | final mean / MAP error | mass within 500 m | mass within 100 m |
|---|---|---:|---:|---:|
| `charles_river_20260727` | forward | 761.04 / 752.73 m | 23.313902% | 0.000662% |
| `charles_river_20260727` | backward | 430.93 / 435.81 m | 48.756264% | 0.062056% |
| `boston_harbor_leg1` | forward | 4,670.10 / 4,753.72 m | 15.798097% | 0.218312% |
| `boston_harbor_leg1` | backward | 357.93 / 269.39 m | 13.724039% | 0.222796% |
| `boston_harbor_leg2` | forward | 7,828.39 / 7,812.28 m | 0.353979% | 0.042518% |
| `boston_harbor_leg2` | backward | 7,797.25 / 7,791.61 m | 0.392694% | 0.302908% |
| `boston_harbor_leg3` | forward | 248.96 / 257.84 m | 14.770582% | 0.000928% |
| `boston_harbor_leg3` | backward | 8,697.36 / 8,536.51 m | 18.317797% | 0.000738% |
| `mount_washington_20260815_leg1` | forward | 1,782.01 / 1,780.54 m | 0.056297% | 0.004096% |
| `mount_washington_20260815_leg1` | backward | 1,704.16 / 1,696.83 m | 0.017863% | 0.002215% |
| `mount_washington_20260815_leg2` | forward | 1,111.59 / 1,107.89 m | 4.282085% | 4.050082% |
| `mount_washington_20260815_leg2` | backward | 1,720.32 / 1,725.46 m | 0.224777% | 0.006541% |
| `mount_washington_20260815_leg3` | forward | 60.53 / 64.94 m | 78.430271% | 34.721764% |
| `mount_washington_20260815_leg3` | backward | 455.76 / 452.01 m | 99.311799% | 41.014407% |

The convergence cost is the path integral of missing posterior probability
mass, `sum((1 - mass_within_radius) * distance_increment)`; it is measured in
meters and lower is better. It is not a threshold-crossing distance.
The table below preserves the released v1 LOCI diagnostic, including its
rounded filename-coordinate truth; it is not the canonical far-field score
reported above.

| leg | mean cost, 25 m | mean cost, 50 m | mean cost, 100 m |
|---|---:|---:|---:|
| `charles_river_20260727` | 5,472.41 m | 5,472.40 m | 5,471.21 m |
| `boston_harbor_leg1` | 13,301.03 m | 13,296.71 m | 13,273.22 m |
| `boston_harbor_leg2` | 6,427.82 m | 6,426.54 m | 6,418.00 m |
| `boston_harbor_leg3` | 18,234.03 m | 18,234.02 m | 18,233.88 m |
| `mount_washington_20260815_leg1` | 392.34 m | 392.34 m | 392.33 m |
| `mount_washington_20260815_leg2` | 792.03 m | 784.85 m | 778.21 m |
| `mount_washington_20260815_leg3` | 1,095.66 m | 906.63 m | 733.99 m |

These are zero-shot baseline results, not a claim that every leg localized
successfully. They show large failures and direction sensitivity alongside a
few low-error directions. In particular, Boston leg 2 remains about 7.8 km
wrong in both directions, while Mount Washington leg 3 ends at 61/455 m
weighted-mean error. Treat the completed artifacts as a reproducible baseline
for diagnosis or later held-out calibration, not as a uniformly successful
system.

#### Manifest digest inventory

These are SHA-256 digests of the exact `manifest.json` files, distinct from
the manifests' internal payload `content_digest` fields.

| shared/reusable artifact | manifest SHA-256 |
|---|---|
| 768-D tag-value dictionary | `969c6c35a1f5e967441e01a06b1b279c65c5bee969134256bfd853bb98b4fa46` |
| Charles satellite WAG embeddings and leg matrix | `157899443a816e498669bad328d83a191abb9e52b68e194af3f06d17fded8c20` |
| Boston shared satellite WAG embeddings | `cd1d519efc2e49c4a025425764f1c10ecabd8cbe59bae7b12be5af7e566d0802` |
| Washington shared satellite WAG embeddings | `3ef4ecf7c24a2e4873a310c8fdf843146702db91b7939e69baf20a2c0ab0283b` |

| leg | correspondence | visual WAG | paths | fused run |
|---|---|---|---|---|
| `charles_river_20260727` | `8e0818f0dfddc1e134924b7a652074ccd58b13212632718a15d5a5e97728ac43` | `157899443a816e498669bad328d83a191abb9e52b68e194af3f06d17fded8c20` | `69de9648a55117daed9b557bfedfdf57b92735dd1d3e58617c4aefcd9e5c9635` | `26e00539ab0a1ef759a318875f319703a7532a58edf1bb2cfbc956b1bff4415f` |
| `boston_harbor_leg1` | `3adb66074f42bf8e46d846af1f788b571c6bbdd5a0b4e52cba4bc963182c0a8d` | `4b754ff551c74da77bf1cd9a9659030097e57d1227c62f45fcb1d69473299862` | `6532edc2834f579ebc110528f301ee2d32708d598a8e4e9543224d8640b6d040` | `cdde21a91258d03b6a6df63d90a077e94693d2f0a5a27a2f6a4a03f6a5de4385` |
| `boston_harbor_leg2` | `01aeacc83254993fedf25ea727db7303e55fb2cf163f3aa46905b60eddf1e336` | `5eb2dd3073af1fdcf869f92fb81686b2a7eeab1d856f7bbf414922cf0b3431c7` | `4982c7806707090e4f109ea7fa8b5a3cd516e8b42a8079de33bad04996fa5154` | `7375c756323703722cbe5bc5abca0f24aa937ea07f4bffa94bb7d98777e4c001` |
| `boston_harbor_leg3` | `39b8accd122c8152943c9ccf977b36215995b3cbe09fefbfbabeceb7308b7033` | `0e8608b7216b4093750ca8ad2a1c363fe8db31df0dfacf068e8ab3442097cb0d` | `ecb0e32c710a4000ea3efa0856db0dea12e01460f8add52162f0b7d4128ff91a` | `86a48407f0aa2675e332f3219ac709464748d5746db529eb2aba9b3f1a64699f` |
| `mount_washington_20260815_leg1` | `9d6e4b7e948201266df01cf93ca1173ee529f9e68d818aab06a9a8318f77ec87` | `ca5d1283f0538570b8abca45148d9ca01fd60af034820b3f89c4435a1ece9481` | `90b0a20d1f46d950c197703c36443c1a9a590172ea5ec34d58d4bf993c8060e6` | `927b1a6b992a62dacbffc124bfa422c93639891c3a58f08cc38c5384df490cbc` |
| `mount_washington_20260815_leg2` | `3637c753d525f2e439886a35b89d2c6ddcd17897f7c9db4e5596750ad2765303` | `dd277fc6d8b6cfb5b8603312cafc4eed7eb1425ce65c021fd15d142c0463495f` | `cb20d79efc32a7a13ab7828e0a8548dfdd688d4c9ffb86985c076e0efa2bd066` | `784f0eb5092ffbc2fa5493922a381a1de322d7428755dcdefb0fd797450d010c` |
| `mount_washington_20260815_leg3` | `31e2a0de542e5ae7469fe75aa79ccce6a901628d97ca7c3d1c3361c01475ec99` | `4ae79f788461f993c2af589ae5e0ee8db8158cf476e70ef165f59ce378247143` | `63916deb5a4e0e6f9b95cfa870b8794ea9b7b319c1f5b239c06f2ec39a7dded6` | `4d4fd652b23134fb41d5db78a434b12b4b9483e1973b5d84e7127dbbbdc07569` |

| leg | canonical 100/500 m mapping-truth replay (v3) |
|---|---|
| `charles_river_20260727` | `c3e57eb3da4e7bb0afcc7a25376dcfd3bd65b09e1d0ec49386fedae783488e1f` |
| `boston_harbor_leg1` | `0e2cb16e8459d1f602600047281364a2a33309d168ee572ddbc19974b776a00e` |
| `boston_harbor_leg2` | `7a7a6d65765e9d03ed62b7ab5bbf9f28de77443545bbcab45727287f18d68cdc` |
| `boston_harbor_leg3` | `be05db63edcd5be22c932492e43f72c415f4b7cc862b12c0c5e7fceaaf25843b` |
| `mount_washington_20260815_leg1` | `1c47ba16ccb8540555b8eb7c7a17652f52bce39774e8b8e5757e67a5ed5b3f9e` |
| `mount_washington_20260815_leg2` | `4e389f2888c0ff2de67878caf2a6ce4a19f61a30d5f1522a46ed495684a9b2f9` |
| `mount_washington_20260815_leg3` | `c5649e20169030185f598734535b7385886d7cab91e95a479d7d23674eb951a2` |

The original v1 audit reopened all 31 intended typed artifacts, matched 67
upstream references and 90 recorded source/model/VLM/dataset hashes to live
content, and checked the deep tensor semantics. The canonical metric replay
reopened its seven additional v3 artifacts, verified bytewise v2 parity for
all filter-state histories, recomputed evaluation-only outputs from mapping
truth, and found no remaining v3 `.incomplete` directory. The superseded v2
metric artifacts remain immutable for provenance and should not be reported.
The sealed v1 payloads retain a few metadata-only legacy values:
every `paths.json` has the placeholder `dataset_hash: "new"` although its typed
manifest binds the exact mapping hash, and the Charles, Boston-leg-1, and
Washington-leg-1
`similarity.json` files name their first-compute `.incomplete` cache path even
though their manifests bind the live cache. Charles and Washington leg 1
correspondence manifests also predate explicit ordered-ID hash fields; their
alignment was audited directly. All seven sealed semantic and visual matrices
predate the new machine-checked matrix-identity sidecars, so the checked-in
configs make their legacy status explicit. The current path and WAG generators
correct the first two metadata fields, and both matrix exporters now write the
ordered identity contract automatically. The sealed raw correspondence files
also predate the new source/alignment identity; their typed manifests and deep
content audit remain authoritative. Immutable v1 directories should not be
rewritten.

## Required provenance and validation

Every new artifact must answer both "what made this?" and "is it aligned with
the things that consume it?" Record:

- immutable artifact kind, dataset/scope, version, completion state, creation
  time, and generator target;
- repository commit and dirty diff/hash;
- exact upstream paths, manifest/content digests, and model/config hashes;
- result-shaping arguments, including embedding model/task/dimension, matching
  method/threshold/weighting, sigmas, and filter seeds/noise;
- ordered panorama-ID and satellite-patch-ID lists or their collision-resistant
  digests, not only matrix shape;
- every output's relative path, byte size, SHA-256, dtype, shape, and finite/NaN
  statistics;
- source imagery provider/version and grid parameters through the referenced
  satellite manifest.

Write into a `.incomplete` sibling and publish the final immutable directory
only after validation. A failed or interrupted job may resume its private
staging data, but directory existence alone never means success. Never rewrite
a completed version; choose a new version name when any scientific input or
result-shaping setting changes.

## Execution order

Use this order to expose errors cheaply before the largest job:

1. Run the seven input preflights.
2. Build and audit the shared 768-D dictionary once.
3. Run Mount Washington leg 1 semantic and WAG export as the smallest complete
   smoke test.
4. Generate its two paths and complete one fused filter run.
5. Run Charles with streaming correspondence output; it is the largest raw
   semantic job.
6. Run the other five legs, reusing the Boston and Washington satellite WAG
   embeddings.
7. Audit all seven result manifests and summarize per-leg/per-direction metrics.

Semantic and visual branches for different legs may run concurrently after
their shared inputs exist. Do not concurrently create one shared dictionary or
one shared satellite-embedding file.

## Status

Status below is a snapshot from 2026-09-02. Treat artifact validation, not this
table, as the completion authority.

- [x] Seven frozen panorama datasets and canonical `pano_id_mapping.csv` files.
- [x] Three trajectory-containing approximately 150 km² regions.
- [x] Three complete z20 satellite artifacts with audited imagery coverage.
- [x] Three LOCI-pruned OSM artifacts.
- [x] Seven complete `gemini-3-flash-preview` VLM annotation results.
- [x] Seven complete 1,536-D `gemini-embedding-001` pano-v2 pickles.
- [x] Artifact-native downstream adapters and focused tests.
- [x] Seven checked-in, paper-sigma LOCI aggregator configs.
- [x] Shared 768-D `text-embedding-005` dictionary: 217,964 entries, zero missing, finite float32, base preserved exactly.
- [x] Seven semantic correspondence artifacts.
- [x] Three scope-level satellite WAG embedding artifacts and seven visual matrices.
- [x] Seven typed forward/reverse path artifacts, source-image mappings and trajectory distances audited.
- [x] Seven fused histogram-filter runs with both directions and retained intermediate states.
- [x] Cross-leg manifest, tensor, alignment, probability-mass, and result audit.

## Authoritative implementation references

The released pipeline remains the algorithmic authority. This document adapts
its layout, not its scientific settings. Paths below are relative to the
`loci-release` checkout:

- `docs/landmark_extraction.md`
- `docs/correspondence_model.md`
- `docs/evaluation.md`
- `docs/reproducing_results.md`
- `paper/pipelines/wag_osm_loci_eval_pipeline.sh`
- `paper/configs/eval/loci/`

Within this repository, the command-line implementations are:

- `experimental/overhead_matching/swag/scripts/precompute_value_embeddings.py`
- `experimental/overhead_matching/swag/scripts/export_correspondence_similarity.py`
- `experimental/overhead_matching/swag/scripts/export_similarity_matrix.py`
- `experimental/overhead_matching/swag/scripts/create_eval_paths_from_panorama_trajectory.py`
- `experimental/overhead_matching/swag/scripts/evaluate_histogram_on_paths.py`
- `experimental/overhead_matching/swag/filter/adaptive_aggregators.py`
