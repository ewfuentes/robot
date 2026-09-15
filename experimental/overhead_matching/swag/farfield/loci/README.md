# Far-field LOCI inputs

This package produces the spatial and panorama inputs consumed by the
[far-field LOCI pipeline](../../../../../docs/farfield/loci_pipeline.md):

1. `loci_regions`: the search region and exact Web-Mercator lattice.
2. `loci_osm_landmarks`: LOCI-pruned OSM geometry intersecting the full patch
   footprint.
3. `loci_satellite`: imagery patches, source and patch hashes, coverage
   evidence, and a contact sheet.
4. `loci_vlm_annotations`: audited LOCI VLM requests, predictions, and derived
   panorama tags.

Do not maintain a dataset inventory here. The manifests under
`/data/farfield_matching/artifacts` record the active versions, exact inputs,
provider pins, bounds, counts, and hashes.

## Region ownership

The region is the sole owner of spatial decisions. By default it insets the
complete catalog bbox equally in metric distance on all four sides until it
reaches 150 km2, while retaining every selected trajectory with at least 500 m
clearance. If containment prevents the requested trim, it records the larger
area instead of cropping a trajectory.

Its persisted grid distinguishes the requested bbox, patch-center bbox, and
complete source-pixel footprint. OSM and satellite producers consume that grid
directly; they must not independently reconstruct or trim the bbox.

The OSM producer requires a complete catalog artifact. It selects
LOCI-vocabulary OSM features with `geometry.intersects` against the full patch
footprint, so boundary-crossing lines and polygons remain whole. It does not
apply any additional semantic-region trim. The catalog's recorded coverage
must contain the complete footprint.

## Generic spatial build

Choose immutable version labels and pass every trajectory sharing the map to
the region producer. Repeat `--trajectory_dataset` once per leg:

```bash
bazel run //experimental/overhead_matching/swag/farfield/loci:region -- \
  --farfield_root /data/farfield_matching \
  --dataset <scope> \
  --trajectory_dataset <leg> \
  --catalog_dir <complete-catalog-artifact> \
  --catalog_dataset <catalog-owner> \
  --target_area_km2 150 \
  --zoom 20 \
  --version <region-version>

bazel run //experimental/overhead_matching/swag/farfield/loci:osm -- \
  --farfield_root /data/farfield_matching \
  --dataset <scope> \
  --region_dir <published-region-artifact> \
  --catalog_dir <complete-catalog-artifact> \
  --catalog_dataset <catalog-owner> \
  --version <osm-version>
```

For a one-leg scope, the scope, leg, and catalog owner may be the same; omit
`--catalog_dataset` when it equals `--dataset`.

Build satellite imagery from the published region:

```bash
bazel run //experimental/overhead_matching/swag/farfield/loci:satellite -- \
  --farfield_root /data/farfield_matching \
  --dataset <scope> \
  --region_dir <published-region-artifact> \
  --version <satellite-version> \
  <provider-specific source pin and recovery arguments>
```

Use `--audit_only` when supported to prove source coverage before downloading.
Run a full imagery build without an outer timeout. Mutable recovery state lives
under `builds/<scope>/`; an identical command validates and reuses completed
source tiles and patches. Do not change a provider release, catalog query, or
raster/layer pin while resuming because each is part of source identity.

Publication is strict: no final artifact appears until coverage, expected
files, image contents, and hashes validate.

## Imagery-provider notes

The exact URL, release, layer, raster selection, request geometry, and coverage
evidence belong in the satellite manifest. The stable provider-specific
constraints are:

- Massachusetts scopes can use MassGIS orthophotos only when they cover the
  complete patch footprint. Offshore Boston Harbor requires a source with
  complete water-edge coverage, such as a pinned USGS NAIP ImageServer mosaic.
- Mount Washington should remain z20. Substituting z19 doubles each patch's
  ground footprint and changes the WAG input scale. Pinned USGS NAIP has been
  used successfully there.
- Flevoland can use a pinned Esri Wayback release. Record the release identity,
  not merely the Esri endpoint.
- Pohang can use the public municipal WMS. The portal presents the selected
  imagery under UI year `2021`, while the service's internal layer name is
  `pohang_2022_1225cm`. They identify the same source rather than two blended
  vintages; preserve both labels in the manifest.

## Loading external inputs

Derived inputs do not live inside `datasets/<leg>`. Point `VigorDataset` at the
frozen panorama dataset and supply the published satellite and OSM payloads:

```python
from pathlib import Path

from experimental.overhead_matching.swag.data.vigor_dataset import (
    VigorDataset,
    VigorDatasetConfig,
)

root = Path("/data/farfield_matching")
leg = "<leg>"
satellite_artifact = Path("<published-satellite-artifact>")
osm_artifact = Path("<published-osm-artifact>")
dataset = VigorDataset(
    root / "datasets" / leg,
    VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        satellite_dir=satellite_artifact / "satellite",
        landmark_path=osm_artifact / "landmarks.feather",
    ),
)
```

Keep both legacy tensor caches disabled for external artifacts because that
cache namespace does not bind their identity. Load each leg as a separate
`VigorDataset`, even when several legs share the same satellite and OSM paths.

Before inference, validate the handoff:

```bash
bazel run //experimental/overhead_matching/swag/farfield/loci:input_smoke -- \
  --dataset_dir /data/farfield_matching/datasets/<leg> \
  --satellite_artifact <published-satellite-artifact> \
  --osm_artifact <published-osm-artifact>
```

This opens and hashes both artifact trees, loads the real compact OSM catalog,
builds satellite, panorama, and landmark associations, and rejects any
panorama without a satellite association.

## VLM request bundles

Generate the request body with the stock LOCI `panov2_tuned_prompt` settings,
then audit it into the version root:

```bash
DATASET=<leg>
VERSION=<new-pro-annotation-version>
PINHOLES=/data/farfield_matching/artifacts/pinhole_images/$DATASET/<version>
ROOT=/data/farfield_matching/artifacts/loci_vlm_annotations/$DATASET/$VERSION

bazel run //experimental/overhead_matching/swag/model:semantic_landmark_extractor -- \
  create_panorama_sentences \
  --pinhole_dir "$PINHOLES" \
  --output_base "$ROOT/sentence_requests" \
  --prompt_type osm_tags \
  --num_workers 8 \
  --media_resolution MEDIA_RESOLUTION_ULTRA_HIGH \
  --thinking_level HIGH \
  --disable_tqdm

bazel run //experimental/overhead_matching/swag/farfield/loci:vlm_requests -- \
  --dataset_dir "/data/farfield_matching/datasets/$DATASET" \
  --pinhole_dir "$PINHOLES" \
  --request_dir "$ROOT/sentence_requests/panorama_sentence_requests" \
  --output_manifest "$ROOT/request_manifest.json" \
  --model gemini-3.1-pro-preview \
  --generator_disable_tqdm
```

The audit pins the exact system prompt, user prompt, response schema, four
byte-identical 2048 x 2048 JPEGs per panorama, per-image
`MEDIA_RESOLUTION_ULTRA_HIGH`, and thinking `HIGH`. The request JSON itself is
model-agnostic; submit it with the model recorded by the manifest. A model
change requires a new immutable version even when every request byte is
unchanged.

The four yaw-labeled faces are panorama-relative, not compass-relative. This
is valid for LOCI's tag-only correspondence branch, which consumes no landmark
bearing. Do not treat their yaw labels as world headings.

After retrieval, require a complete response for every request key, build the
released pano-v2 annotation payload, and record the response and embedding
manifests beside it. Downstream correspondence export reads the structured
panorama tag bundles; it does not numerically consume the pano-v2 1,536-D
annotation vectors.
