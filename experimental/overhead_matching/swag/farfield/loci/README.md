# Far-field LOCI inputs

This package turns one complete far-field catalog plus its canonical
trajectory tables into four independent, versioned input trees:

1. `loci_regions`: the requested search box and exact Web-Mercator lattice.
2. `loci_osm_landmarks`: LOCI-pruned OSM geometry intersecting the complete
   satellite-patch envelope.
3. `loci_satellite`: imagery patches, exact source/patch hashes, provider
   coverage evidence, and a visual contact sheet.
4. `loci_vlm_annotations`: exact LOCI VLM requests, their source-image and
   prompt bindings, downloaded predictions, and derived panorama tags.

The region, OSM, and satellite trees are typed `farfield.artifact.v1`
artifacts. The VLM tree instead carries its request, execution, and embedding
manifests; the consolidated runbook records their exact hashes as untyped
upstream inputs.

The region is the sole owner of spatial decisions.  The other producers read
its persisted grid rather than recomputing a bbox.  By default the region
producer insets the full-catalog bbox equally in metric distance on all four
sides until it reaches 150 km², while retaining every canonical GPS point with
at least 500 m clearance.  If containment requires more area, it records the
larger result instead of cropping a trajectory.

## Charles River

```bash
bazel run //experimental/overhead_matching/swag/farfield/loci:region -- \
  --farfield_root /data/farfield_matching \
  --dataset charles_river_20260727 \
  --catalog_dir /data/farfield_matching/artifacts/catalogs/charles_river_20260727/stage3_7b88e81_full_v1 \
  --version area150km2_v2

bazel run //experimental/overhead_matching/swag/farfield/loci:osm -- \
  --farfield_root /data/farfield_matching \
  --dataset charles_river_20260727 \
  --region_dir /data/farfield_matching/artifacts/loci_regions/charles_river_20260727/area150km2_v2 \
  --catalog_dir /data/farfield_matching/artifacts/catalogs/charles_river_20260727/stage3_7b88e81_full_v1 \
  --version area150km2_osm260101_v2

bazel run //experimental/overhead_matching/swag/farfield/loci:satellite -- \
  --farfield_root /data/farfield_matching \
  --dataset charles_river_20260727 \
  --region_dir /data/farfield_matching/artifacts/loci_regions/charles_river_20260727/area150km2_v2 \
  --version area150km2_massgis2025_z20_nearestpx_v2 \
  --build_cache_version area150km2_massgis2025_z20_v1
```

The satellite command has no outer timeout.  Its mutable state lives under
`builds/<dataset>/loci_satellite_<build_cache_version>/` (or the artifact
version when `--build_cache_version` is omitted).  Rerunning the same command
checks and reuses every valid source tile and assembler-keyed patch and
atomically replaces corrupt cache entries.  Publication is strict: no artifact
appears until coverage, all source tiles, all patches, and all hashes validate.

`area150km2_massgis2025_z20_v1` is retired under
`artifacts/_retired/20260901_charles_loci_v1/`: its release-compatible
fractional crop left a one-pixel edge in the rendered patches.  Use the
`nearestpx_v2` artifact above, whose crop origin is quantized once to the
nearest source pixel and whose patches have been fully edge-audited.  The
similarly named mutable build cache remains intentionally active because it
contains the shared source tiles and corrected v2 repair patch set.

Use `--audit_only` on the satellite command to run the ArcGIS Tilemap and
source-orthophoto-index coverage proof without downloading imagery.

## Boston Harbor (legs 1--3)

Use one shared artifact scope for all three Boston Harbor legs.  The region
producer must see every trajectory even though the byte-identical leg 1
catalog is the canonical spatial catalog:

```bash
bazel run //experimental/overhead_matching/swag/farfield/loci:region -- \
  --farfield_root /data/farfield_matching \
  --dataset boston_harbor_shared \
  --trajectory_dataset boston_harbor_leg1 \
  --trajectory_dataset boston_harbor_leg2 \
  --trajectory_dataset boston_harbor_leg3 \
  --catalog_dir /data/farfield_matching/artifacts/catalogs/boston_harbor_leg1/stage3_7b88e81_full_v1 \
  --catalog_dataset boston_harbor_leg1 \
  --target_area_km2 150 \
  --version area150km2_v1

bazel run //experimental/overhead_matching/swag/farfield/loci:osm -- \
  --farfield_root /data/farfield_matching \
  --dataset boston_harbor_shared \
  --region_dir /data/farfield_matching/artifacts/loci_regions/boston_harbor_shared/area150km2_v1 \
  --catalog_dir /data/farfield_matching/artifacts/catalogs/boston_harbor_leg1/stage3_7b88e81_full_v1 \
  --catalog_dataset boston_harbor_leg1 \
  --version area150km2_osm260101_v1
```

The published region is containment-limited to 153.770811 km², rather than
cropping any trajectory to hit exactly 150 km².  Its bbox is
`[-71.0538148896331, 42.249740523579405, -70.9121447103669,
42.368180876420595]`; its z20 grid is 331 by 374, or 123,794 patches.  The
shared OSM artifact contains 56,505 landmarks.  The exact artifact roots are:

```text
/data/farfield_matching/artifacts/loci_regions/boston_harbor_shared/area150km2_v1
/data/farfield_matching/artifacts/loci_osm_landmarks/boston_harbor_shared/area150km2_osm260101_v1
```

Use the pinned 2023 USGS NAIP ImageServer export for the shared imagery:

```bash
bazel run //experimental/overhead_matching/swag/farfield/loci:satellite -- \
  --farfield_root /data/farfield_matching \
  --dataset boston_harbor_shared \
  --region_dir /data/farfield_matching/artifacts/loci_regions/boston_harbor_shared/area150km2_v1 \
  --version area150km2_usgsnaip2023_z20_nearestpx_v2 \
  --build_cache_version area150km2_usgsnaip2023_z20_chunk15_v2 \
  --provider_mode arcgis_image_server_export \
  --service_url https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer \
  --catalog_where "State='MA' AND Year=2023 AND Category=1" \
  --lock_raster_id 72904 \
  --lock_raster_id 72905 \
  --lock_raster_id 72906 \
  --lock_raster_id 72907 \
  --lock_raster_id 72911 \
  --lock_raster_id 72912 \
  --lock_raster_id 73072 \
  --lock_raster_id 73074 \
  --lock_raster_id 73104 \
  --image_server_chunk_tiles 15 \
  --workers 2 \
  --max_retries 12
```

The explicit `chunk15` build-cache version is intentionally fresh: it cannot
bind the paused legacy cache that requested one export per source tile.  Chunk
mode partitions the source-tile range into fixed, row-major 15 by 15 blocks
anchored at its northwest corner (with smaller blocks only at the east and
south edges).  Each request is therefore at most 3840 by 3840 pixels, the
ImageServer's advertised maximum.  A response is split into canonical
256-by-256 RGB PNG child tiles; all children are validated and written before
an atomic receipt commits their hashes.  Resume accepts a block only when its
receipt and every child tile still validate.  A missing or invalid receipt
re-exports and rebuilds the whole block, even if some child files remain.

The earlier `area150km2_massgis2025_z20_nearestpx_v1` Boston satellite artifact
is retired under
`artifacts/_retired/20260901_boston_harbor_massgis_nodata_v1/`; do not select
it.  Although its structure and hashes validate, the MassGIS mosaic contains
an offshore no-data wedge (6,051 effectively black patches) in the northeast
of the shared lattice.  The pinned NAIP replacement has complete catalog
coverage of the rendered footprint.

## Mount Washington (legs 1--3)

Mount Washington likewise uses one shared scope and the leg 1 catalog.  Keep
the primary artifact at z20 so its 640 px patches have LOCI's expected ground
footprint; z19 doubles the footprint and is not scale-compatible.

```bash
bazel run //experimental/overhead_matching/swag/farfield/loci:region -- \
  --farfield_root /data/farfield_matching \
  --dataset mount_washington_20260815_shared \
  --trajectory_dataset mount_washington_20260815_leg1 \
  --trajectory_dataset mount_washington_20260815_leg2 \
  --trajectory_dataset mount_washington_20260815_leg3 \
  --catalog_dir /data/farfield_matching/artifacts/catalogs/mount_washington_20260815_leg1/stage3_7b88e81_full_v1 \
  --catalog_dataset mount_washington_20260815_leg1 \
  --target_area_km2 150 \
  --zoom 20 \
  --version area150km2_z20_v2

bazel run //experimental/overhead_matching/swag/farfield/loci:osm -- \
  --farfield_root /data/farfield_matching \
  --dataset mount_washington_20260815_shared \
  --region_dir /data/farfield_matching/artifacts/loci_regions/mount_washington_20260815_shared/area150km2_z20_v2 \
  --catalog_dir /data/farfield_matching/artifacts/catalogs/mount_washington_20260815_leg1/stage3_7b88e81_full_v1 \
  --catalog_dataset mount_washington_20260815_leg1 \
  --version area150km2_z20_osm260101_v2
```

The published region is 150 km² with bbox
`[-71.38860670969237, 44.20986503095438, -71.23484429030763,
44.31979396904561]`.  Its z20 grid is 359 by 358, or 128,522 patches, with a
roughly 68.4 m ground footprint per patch.  The shared OSM artifact contains
2,970 landmarks.  The exact artifact roots are:

```text
/data/farfield_matching/artifacts/loci_regions/mount_washington_20260815_shared/area150km2_z20_v2
/data/farfield_matching/artifacts/loci_osm_landmarks/mount_washington_20260815_shared/area150km2_z20_osm260101_v2
```

The pinned 2023 New Hampshire NAIP catalog covers the full z20 rendered
footprint:

```bash
bazel run //experimental/overhead_matching/swag/farfield/loci:satellite -- \
  --farfield_root /data/farfield_matching \
  --dataset mount_washington_20260815_shared \
  --region_dir /data/farfield_matching/artifacts/loci_regions/mount_washington_20260815_shared/area150km2_z20_v2 \
  --version area150km2_usgsnaip2023_z20_nearestpx_v2 \
  --build_cache_version area150km2_usgsnaip2023_z20_chunk15_v2 \
  --provider_mode arcgis_image_server_export \
  --service_url https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer \
  --catalog_where "State='NH' AND Year=2023 AND Category=1" \
  --lock_raster_id 125295 \
  --lock_raster_id 125297 \
  --lock_raster_id 125299 \
  --lock_raster_id 125300 \
  --lock_raster_id 125301 \
  --lock_raster_id 125302 \
  --lock_raster_id 125304 \
  --lock_raster_id 125306 \
  --lock_raster_id 125327 \
  --lock_raster_id 125331 \
  --lock_raster_id 125332 \
  --lock_raster_id 125336 \
  --image_server_chunk_tiles 15 \
  --workers 2 \
  --max_retries 12
```

The 15 by 15 source-tile chunks stay below the ImageServer's 4,000 px export
limit and commit atomically, so the identical command resumes safely.

For either ImageServer workflow, first adding `--audit_only` performs the
pinned-catalog coverage proof without downloading imagery.  Run the full
satellite command without an outer timeout.  Dynamic exports can occasionally
return transient HTTP 502 responses; `--max_retries 12` handles short
interruptions, and rerunning the identical command resumes valid tiles and
patches from
`builds/<shared-scope>/loci_satellite_<build-cache-version>/`.  Do not change
the WHERE clause or raster IDs while resuming: they are part of the strict
source identity.  The explicit `--build_cache_version` resumes
receipt-committed blocks.

## Loading from the frozen dataset

Derived inputs do not live in `datasets/<dataset>`.  Point `VigorDataset` at
the frozen panorama dataset and provide the two artifact payloads explicitly:

```python
from pathlib import Path
from experimental.overhead_matching.swag.data.vigor_dataset import (
    VigorDataset,
    VigorDatasetConfig,
)

root = Path("/data/farfield_matching")
name = "charles_river_20260727"
config = VigorDatasetConfig(
    satellite_tensor_cache_info=None,
    panorama_tensor_cache_info=None,
    satellite_dir=(root / "artifacts/loci_satellite" / name /
                   "area150km2_massgis2025_z20_nearestpx_v2/satellite"),
    landmark_path=(root / "artifacts/loci_osm_landmarks" / name /
                   "area150km2_osm260101_v2/landmarks.feather"),
)
dataset = VigorDataset(root / "datasets" / name, config)
```

Keep both tensor-cache fields disabled for external artifacts.  The legacy
cache namespace does not include artifact identity, so the adapter rejects
external inputs with caches enabled rather than silently loading tensors made
from different imagery or landmarks.

The corrected Mount Washington satellite artifact is z20, matching
`VigorDatasetConfig`'s default; do not override it to z19.  The `input_smoke`
command reads the z20 zoom from the satellite manifest automatically.
Load each leg as a separate `VigorDataset` while pointing all three legs at
the same shared satellite and OSM payloads.

## VLM request bundles

Use the stock LOCI request generator with the production `panov2_tuned_prompt`
settings, then publish its validation manifest beside the request shard:

```bash
DATASET=charles_river_20260727
VERSION=panov2_tuned_prompt_gemini3flash_2048_ultrahigh_v1
PINHOLES=/data/farfield_matching/artifacts/pinhole_images/$DATASET/stage3_7b88e81_adopted_v2
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
  --generator_disable_tqdm
```

The manifest pins the exact production prompt, user prompt, response schema,
four byte-identical 2048 px JPEGs per request, per-image
`MEDIA_RESOLUTION_ULTRA_HIGH`, and thinking `HIGH`.  The request itself does
not name a model; submit it as `gemini-3-flash-preview` as recorded by the
manifest.

The four faces are camera-relative because the pinhole renderer uses fixed
panorama-relative yaw offsets and does not apply per-frame heading metadata,
even when the dataset carries it.  These bundles are valid for the current LOCI
OSM-tag extraction and tag-only correspondence late fusion, which does not
consume yaw.  Do not use their yaw metadata as world bearings or as a
compass-faithful full-LOCI annotation contract.

Before a run, exercise that same handoff against the complete published
artifacts.  This validates both artifact trees, loads the real compact OSM
catalog, builds all satellite/panorama/landmark correspondences, and fails if
any panorama has no satellite association:

```bash
bazel run //experimental/overhead_matching/swag/farfield/loci:input_smoke -- \
  --dataset_dir /data/farfield_matching/datasets/charles_river_20260727 \
  --satellite_artifact /data/farfield_matching/artifacts/loci_satellite/charles_river_20260727/area150km2_massgis2025_z20_nearestpx_v2 \
  --osm_artifact /data/farfield_matching/artifacts/loci_osm_landmarks/charles_river_20260727/area150km2_osm260101_v2
```

For another scope, select its complete catalog and canonical trajectory
dataset(s), choose new immutable version labels, and repeat the same three
commands.  Pass `--trajectory_dataset` more than once when several legs share
one search region.
