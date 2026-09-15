# Far-field LOCI pipeline

This is the start-to-finish contract for applying the released LOCI
late-fusion pipeline to a far-field dataset. It records the stable scientific
choices, reuse boundaries, commands, and validation requirements. It does not
inventory completed runs: the manifests under
`/data/farfield_matching/artifacts` are the source of truth for active versions,
inputs, counts, hashes, and completion state.

The [LOCI input guide](../../experimental/overhead_matching/swag/farfield/loci/README.md)
describes region, OSM, satellite, and VLM input production. The separate
[far-field pipeline](pipeline.md) describes the bearing-based localization
system.

## Pipeline and reuse boundaries

```text
complete catalog + all scope trajectories
                  |
                  v
       ~150 km2 region and z20 grid
           /                 \
          v                   v
LOCI-pruned OSM          satellite patches
          |                   |
panorama VLM tags        released WAG
          |                   |
          v                   v
semantic correspondence   visual similarity
          |                   |
          +--------+----------+
                   v
             LOCI late fusion
                   |
     one complete forward trajectory
                   |
                   v
          histogram Bayes filter
                   |
                   v
       beliefs, errors, 100/500 m mass
```

Artifacts are shared only when their inputs are genuinely shared:

- Region, satellite, OSM, and satellite WAG embeddings are per geographic
  scope. Several recorded legs may point to one scope.
- The 768-D tag-value dictionary can be shared across every leg whose complete
  OSM and panorama tag vocabulary it covers.
- VLM annotations, raw correspondence scores, panorama WAG embeddings, final
  similarity matrices, paths, and filter results are per leg.

Do not run two jobs that create the same shared dictionary or satellite WAG
cache concurrently. Build and validate it once, then let dependent legs read
the published artifact.

## Spatial input contract

The region artifact owns every spatial decision. Starting from a complete OSM
catalog bbox, the producer insets all four sides by the same metric distance
until the region is about 150 km2. It caps that inset rather than excluding any
canonical trajectory point or violating the configured trajectory clearance.
The resulting area may therefore be larger than the target.

The region also records the exact Web-Mercator lattice. Distinguish these
bounds:

- `bbox_wsen` is the requested search region.
- `grid.center_bbox_wsen` spans the first and last patch centers.
- `grid.footprint_bbox_wsen` includes each patch's complete source-pixel
  footprint and is the coverage boundary for both imagery and OSM.

The OSM producer reads the complete catalog and retains LOCI-vocabulary OSM
geometries that intersect the full patch footprint. It does not apply a second
implicit trim, and crossing lines or polygons remain whole. The selected
catalog must cover that footprint according to its own manifest. The satellite
producer must likewise prove complete source coverage of the footprint before
publication.

Use the persisted region grid everywhere downstream; do not reconstruct a bbox
or lattice from a dataset name.

## Panorama VLM contract

New runs use `gemini-3.1-pro-preview` with the stock LOCI
`panov2_tuned_prompt` request:

- the exact pinned system prompt, user prompt, and response schema;
- four byte-identical 2048 x 2048 JPEG pinholes at panorama-relative yaw
  offsets 0, 90, 180, and 270 degrees;
- per-image `MEDIA_RESOLUTION_ULTRA_HIGH`;
- thinking level `HIGH`.

The serialized request JSON is model-agnostic. Pass
`--model gemini-3.1-pro-preview` to `farfield/loci:vlm_requests` so the request
manifest binds the intended submission model, and submit the bundle with that
same model. Changing only the model still requires a new immutable annotation
version.

The faces are camera-relative: the pinhole renderer does not apply per-frame
world-heading metadata. That is sufficient for the tag-only LOCI semantic
branch, which consumes no landmark bearing. Do not reinterpret face yaw as a
world bearing.

## Embedding and late-fusion contracts

There are two unrelated embedding products:

| embedding | model and dimension | use |
|---|---|---|
| VLM annotation embedding | `gemini-embedding-001`, 1,536-D, `SEMANTIC_SIMILARITY` | Preserves the released pano-v2 payload. The correspondence exporter reads its structured `panoramas` tag bundles; the 1,536-D vectors themselves are not classifier inputs. |
| correspondence tag-value embedding | `text-embedding-005`, 768-D, `SEMANTIC_SIMILARITY` | Supplies the released correspondence classifier's embedding lookup for every recognized OSM and VLM tag value. |

Never use `--allow_missing_text_embeddings` for a scientific run. Missing
values must be embedded and added to a new shared 768-D dictionary artifact.

The semantic exporter computes raw landmark-pair `P(match)` scores and reduces
them to a panorama x satellite matrix with Hungarian matching, summation, a
0.8 probability threshold, and panorama-landmark uniqueness weighting. LOCI
does use this semantic matrix: the final observation likelihood is late fusion
of it with the WAG image-similarity matrix. An all-zero or constant semantic row
falls back to image-only evidence through the normalized aggregator.

## Frozen checkpoints and fusion settings

Use the released/local weights below; do not retrain or silently substitute
another checkpoint.

| role | path | identity |
|---|---|---|
| correspondence classifier | `/data/overhead_matching/hf_release_staging/checkpoints/correspondence_classifier/best_model.pt` | SHA-256 `1d45cb6b04f2edfd847160f90c70ad9b2249bb7a192401eb815f9179833d84c3`; byte-identical to local `simple_v1_v5/best_model.pt` |
| WAG, full training output | `/data/overhead_matching/training_outputs/260215_baseline_retraining/260421_221726_all_chicago_dinov3_wag_bs18_v2_no_hinge` | Matches the released paper WAG config except for output-directory paths; `best_panorama/model.pt` SHA-256 `bc24f9401797617d90d6fe3c4fe8bb58673dc8c59dac1f337e166e3a5e24e880`, `best_satellite/model.pt` SHA-256 `ad760c668015faf4a4884d30c614276425c340557fdd78f753f4e74355baa3a2` |
| base tag dictionary | `/data/overhead_matching/hf_release_staging/correspondence/text_value_embeddings.pkl` | Released 206,277-entry `text-embedding-005` table; SHA-256 `e42b103296390b3724fab8dc3410ca345ef550a7a46289baea5e3a0b991e580d` |

Use the paper-frozen late-fusion sigmas for the zero-shot baseline:

```yaml
kind: SafaPlusNormalizedLandmarkAggregatorConfig
image_similarity_matrix_path: <leg WAG similarity.pt>
landmark_similarity_matrix_path: <leg Hungarian similarity.pt>
image_sigma: 0.1809
landmark_sigma: 0.4673
landmark_use_raw_residual: false
```

Do not tune either sigma on a target leg and report the result as zero-shot.

## Evaluation protocol

The comparable LOCI run uses:

- a uniform prior over the complete rendered support;
- one path containing the full recorded trajectory in canonical
  `pano_id_mapping.csv` order, with no reverse copy;
- LOCI's existing known-world-heading motion model;
- the same realized translational odometry used by the corresponding
  far-field run, projected into LOCI's world-frame delta representation;
- exact applied controls persisted as `applied_motion_deltas.pt` in each path
  result and loaded by every replay or step-through viewer;
- posterior probability mass within 100 m and 500 m of mapping-table truth.

The persisted controls, not a seed and reimplementation of the noise sampler,
are the replay authority. Record the source odometry artifact identity and the
projection convention in the run manifest. Keep filter process uncertainty
separate from the already-realized input odometry corruption.

LOCI retains a known-heading advantage over the heading-marginalized far-field
filter. The shared translational realization makes position input comparable;
it does not claim to make the state spaces identical.

Probability mass is sampled after each panorama observation, excluding the
initial uniform prior. The canonical score is the trapezoidal area under that
mass curve over true distance travelled, divided by total trajectory length,
as implemented in
`experimental/overhead_matching/swag/farfield/localization/metrics.py`. The
100 m and 500 m values are two cumulative-radius metrics, not a 100--500 m
annulus.

## Artifact layout

Choose immutable, descriptive version names beneath these lanes:

```text
/data/farfield_matching/artifacts/
  loci_regions/<scope>/<version>/
  loci_satellite/<scope>/<version>/
  loci_osm_landmarks/<scope>/<version>/
  loci_vlm_annotations/<leg>/<version>/
  loci_text_value_embeddings/<coverage-set>/<version>/
  loci_correspondence_scores/<leg>/<version>/
  loci_wag_similarity/<scope-or-leg>/<version>/
  loci_eval_paths/<leg>/<version>/
  loci_runs/<leg>/<version>/
```

A current semantic or visual matrix has a JSON identity sidecar binding its
ordered panorama IDs and satellite patch IDs. A current path binds the exact
`pano_id_mapping.csv` digest. Strict consumers require those identities and
reject mismatches even when tensor shapes agree.

`allow_legacy_similarity_identity` and `--allow-legacy-path-identity` exist
only to inspect historical artifacts that predate these contracts. Do not add
them to a new configuration or run. Regenerate obsolete path or matrix
artifacts instead of making a current baseline depend on either escape hatch;
after validated replacements exist, remove the obsolete paths and their
dependent runs together.

## Procedure for a new dataset

1. Choose the complete catalog and every trajectory that will share its map.
   Generate the approximately 150 km2 region and its z20 lattice.
2. Generate and validate the footprint-complete OSM and satellite artifacts.
3. Render or select the adopted 2048 px pinholes. Generate the exact LOCI
   request bundle, bind the Pro model in its manifest, submit it, validate all
   responses, and build the pano-v2 annotation payload.
4. Extend the shared 768-D tag dictionary for any new recognized values.
5. Export the semantic correspondence matrix and its strict ordered-identity
   sidecar. Stream the raw score matrix when its size warrants it.
6. Export satellite WAG embeddings once per scope and panorama WAG similarity
   once per leg, with strict ordered identity.
7. Create the full recorded path with
   `create_eval_paths_from_panorama_trajectory --full_trajectory --forward-only`.
8. Convert the matching far-field odometry artifact into LOCI world-frame
   deltas, run the late-fusion histogram filter, and retain the exact applied
   deltas with the result.
9. Compute the canonical distance-normalized 100 m and 500 m mass summaries
   and validate the complete artifact lineage before publication.

Run `farfield/loci:input_smoke` before the inference stages. It opens and
hashes the map artifacts, builds the real `VigorDataset`, checks the lattice,
and requires a satellite association for every panorama.

Semantic and visual exports for independent legs may run concurrently after
their shared inputs exist. Run the smallest leg through the entire pipeline
before starting the largest raw correspondence export.

## Required provenance and validation

Every new artifact must answer both "what made this?" and "is it aligned with
the things that consume it?" Record:

- immutable artifact kind, dataset or scope, version, completion state,
  creation time, and generator target;
- repository commit and dirty diff or digest;
- exact upstream paths, manifest/content digests, and model/config hashes;
- result-shaping arguments, including VLM and embedding model, task and
  dimension, matching policy, sigmas, path direction, and filter settings;
- source imagery provider/version and grid parameters through the satellite
  manifest;
- ordered panorama and satellite identities, not only matrix shape;
- every output's relative path, byte size, SHA-256, dtype, shape, and
  finite/NaN statistics;
- source odometry identity and the exact persisted applied controls.

Write into a `.incomplete` sibling and publish the final immutable directory
only after content, coverage, and alignment validation. Directory existence or
a successful process exit is not proof of completion. Never rewrite a
completed version; choose a new version when a scientific input or
result-shaping setting changes.

At minimum, validate:

- every trajectory lies within the region with its promised clearance;
- the complete catalog and imagery source cover the full patch footprint;
- every expected satellite patch and panorama is present exactly once;
- VLM request and response keys exactly match the mapping table;
- the tag dictionary covers every recognized value with finite float32 768-D
  vectors while preserving all base entries;
- semantic and WAG matrices are finite and bind the same ordered IDs used by
  the filter;
- every belief is finite, nonnegative, and normalized;
- 500 m posterior mass is pointwise at least 100 m mass;
- the recorded mass summaries reproduce a fresh call to the canonical metric.

## Imagery-provider notes

Keep provider-specific recovery settings and source pins in each satellite
manifest rather than copying them into this runbook.

- Massachusetts scopes have used MassGIS orthophotos where their complete
  patch footprint is covered. Offshore Boston Harbor required a pinned USGS
  NAIP ImageServer mosaic instead.
- Mount Washington uses pinned USGS NAIP at z20. Do not substitute z19: it
  doubles the ground footprint and changes WAG input scale.
- Flevoland uses a pinned Esri Wayback release; the release identifier belongs
  in the source manifest.
- Pohang uses the public municipal WMS. The portal displays the selected source
  under UI year `2021`, while the service's internal layer name is
  `pohang_2022_1225cm`. These identify the same selected imagery, not blended
  sources, and both labels should be retained in provenance.

## Authoritative implementation references

The released pipeline remains the algorithmic authority. Paths below are
relative to the `loci-release` checkout:

- `docs/landmark_extraction.md`
- `docs/correspondence_model.md`
- `docs/evaluation.md`
- `docs/reproducing_results.md`
- `paper/pipelines/wag_osm_loci_eval_pipeline.sh`
- `paper/configs/eval/loci/`

Within this repository, the command-line implementations are:

- `experimental/overhead_matching/swag/farfield/loci/region.py`
- `experimental/overhead_matching/swag/farfield/loci/osm.py`
- `experimental/overhead_matching/swag/farfield/loci/satellite.py`
- `experimental/overhead_matching/swag/farfield/loci/vlm_requests.py`
- `experimental/overhead_matching/swag/scripts/precompute_value_embeddings.py`
- `experimental/overhead_matching/swag/scripts/export_correspondence_similarity.py`
- `experimental/overhead_matching/swag/scripts/export_similarity_matrix.py`
- `experimental/overhead_matching/swag/scripts/create_eval_paths_from_panorama_trajectory.py`
- `experimental/overhead_matching/swag/scripts/evaluate_histogram_on_paths.py`
- `experimental/overhead_matching/swag/scripts/summarize_loci_position_mass.py`
- `experimental/overhead_matching/swag/filter/adaptive_aggregators.py`
