# CrossLocate-Depth full-dataset handoff

This is the handoff for building the CrossLocate-Depth reference databases and
scoring every frozen far-field panorama sequence. The runner owns elevation
source acquisition, surface construction, database rendering, and scoring. The
downstream owner will build the current `localization_inputs` artifacts and run
the particle filter after the score fields come back.

## Start here

Use branch `baseline-data` (PR #700). It contains the retrieval/filter work
from `cld3-retrieval-likelihood` (PR #692) and was restacked on
`farfield-crossview-base` at `5c922921` on 2026-09-11. Do not start from the
old `dem_baseline` branch or from an existing rendered database.

```bash
git fetch origin
git switch --track origin/baseline-data
export FARFIELD_ROOT=/absolute/path/to/incoming
```

The handed-off tree is the `incoming/` directory shared out of band. It can be
used directly as `FARFIELD_ROOT`; it already follows the repository's data
layout.

## Division of work

The runner should:

1. Download the public pre-gridded elevation products listed below.
2. Build or reproject height fields covering the declared candidate region and
   its 30 km render halo. Do not rebuild either Massachusetts DSM from LiDAR.
3. Build the reference descriptor database(s).
4. Score every panorama in every sequence with `--keyframe_stride 1`.
5. Return the retrieval fields and the small provenance manifests listed under
   “Output contract.”

The runner does **not** need to build landmark detections, matches,
`localization_inputs`, or filter runs. Score artifacts use the frozen dataset's
`frame_idx` and panorama IDs, so they can be joined to a later
`localization_inputs` build from the same dataset bytes.

## Copied data

The handoff contains all inputs that were expensive, private, or easy to get
wrong:

```text
incoming/
  datasets/<13 dataset names>/
  artifacts/catalogs/<each dataset>/<final version>/
  artifacts/dem_surfaces/boston_harbor/v1_dsm/
  artifacts/dem_surfaces/boston_harbor/AUDIT.md
  artifacts/dem_surfaces/charles_river/v1_dsm/
  artifacts/dem_surfaces/charles_river/AUDIT.md
  models/crosslocate/AlpsPhotosToDepthCompact_31_2/
```

The data payload is 9,230 regular files / 45.064 GiB, plus this README. Of
that, the datasets are 9,193 files / 41.936 GiB. Each dataset also has a
`panorama -> frames` symlink so the scoring CLI resolves it without changing
the frozen images. The 13 final catalog directories contain each dataset's own
manifest, even where several legs share byte-identical catalog rows.

Only these model files were copied:

- `converted_weights.npz`
- `converted_weights.npz.manifest.json`
- `port_verification.json`

Each copied Massachusetts DSM contains `surface.npz`, `surface.json`, and
`provenance.npz`. These are the completed 1 m USGS 2021 QL1 LiDAR-derived
surfaces; the raw point clouds are deliberately not part of the handoff.

The public source tiles, coarse GLO-30 backgrounds, existing rendered
databases, retrieval observations, filter inputs, and old runs were not
copied. In particular, do not reuse the existing Boston/Charles databases:
they predate the final 625 km² regions. The transferred files were checked
against the source by size; the SFTP endpoint does not expose a common content
hash.

## Dataset and catalog roster

`manifest.json:config.region_bbox_wsen` in the named catalog is the sole source
of truth for candidate support. Do not derive support from the trajectory or
from the extent of catalog rows.

| Dataset(s) | Frames | Final catalog version |
|---|---:|---|
| `mount_washington_20260815_leg1` | 128 | `trim_20260910_v1` |
| `mount_washington_20260815_leg2` | 265 | `trim_20260910_v1` |
| `mount_washington_20260815_leg3` | 272 | `trim_20260910_v1` |
| `pohang_canal_04` | 1,450 | `trim_20260910_v1` |
| `flevoland_polder` | 633 | `trim625_20260910_v1` |
| `charles_river_20260727` | 513 | `trim625_20260910_v1` |
| `boston_harbor_leg1` | 379 | `trim625_20260911_v1` |
| `boston_harbor_leg2` | 236 | `trim625_20260911_v1` |
| `boston_harbor_leg3` | 734 | `trim625_20260911_v1` |
| `franconia_leg1` | 808 | `trim625_20260910_v1` |
| `portland_flight_20260906_leg1` | 492 | `trim_osmfaa_20260910_v1` |
| `portland_flight_20260906_leg2` | 507 | `trim_osmfaa_20260910_v1` |
| `portland_flight_20260906_leg3` | 480 | `trim_osmfaa_20260910_v1` |

## Recommended elevation source by region

Use downloadable rasters, not point clouds, except that the already-built
Massachusetts DSMs are supplied.

| Region / sequences | Primary surface | Coarse fallback / halo | Metric CRS |
|---|---|---|---|
| Mount Washington, all three legs | USGS 3DEP 1/3 arc-second DEM, about 10 m bare earth | None if 3DEP has complete support + halo coverage | EPSG:26919 |
| Pohang | Copernicus DEM GLO-30, 30 m | Same product | EPSG:32652 |
| Flevoland | AHN open-data pre-gridded 5 m DTM raster; record the exact AHN generation | GLO-30 for water, holes, or halo not covered by AHN | EPSG:28992 |
| Charles River | Copied `charles_river/v1_dsm/surface`, 1 m static DSM | GLO-30 over the whole region + 30 km halo | EPSG:6348 |
| Boston Harbor, all three legs | Copied `boston_harbor/v1_dsm/surface`, 1 m static DSM | GLO-30 over the whole region + 30 km halo | EPSG:6348 |
| Franconia | USGS 3DEP 1/3 arc-second DEM, about 10 m bare earth | None if 3DEP has complete support + halo coverage | EPSG:26919 |
| Portland/Maine, all three flight legs | **USGS 3DEP 1/3 arc-second bare-earth DEM** over support + halo | GLO-30 for coastal water/no-data only | EPSG:26919 |

USGS rasters are available through The National Map downloader. For Maine, do
not download LiDAR point clouds and do not build a DSM. GLO-30 is technically
a DSM and uses EGM2008 heights; record that fact and any vertical-datum
mismatch wherever it backs a USGS/AHN foreground.

## Spatial contract: support is not the render halo

The candidate lattice and the terrain required to render it have different
extents:

- **Candidate support:** exactly the final catalog's declared
  `config.region_bbox_wsen`, at 100 m lattice spacing.
- **Terrain coverage:** candidate support plus 30,000 m in every direction in
  the chosen metric CRS.
- **Render range:** 30,000 m. The halo supplies geometry seen from boundary
  candidates; it never adds candidate locations.
- **Filtering later:** `--prior_region catalog --margin_m 0`.

`render_db` currently accepts only a projected `--bounds_xy`; it does not read
or hash the catalog manifest. Before the production render, bind this step to
the catalog: read `region_bbox_wsen`, project all four corners into the surface
CRS, use their min/max as the lattice bounds, and record the catalog manifest's
SHA-256 in the database manifest. This matches the projected-corner convention
used by `localization/build_export.py`. If literal WGS84 rectangle membership
is required rather than that projected bounding box, a node mask must also be
added; the current CLI does not provide one.

The completed database must report `n_dropped_nodata: 0`. Every view's
coverage should be 1.0, including database locations at the region boundary.

## Fixed rendering and scoring settings

Use these settings for every production database:

| Setting | Value |
|---|---:|
| Candidate lattice | 100 m |
| Headings | 12, spaced 30° |
| Perspective view | 60° FOV, 500 × 500 |
| Descriptor | 512-D CrossLocate VGG16-MAC, float16 on disk |
| Maximum range | 30,000 m |
| Sky encoding | `-1` |
| Earth curvature | enabled |
| Refraction coefficient | `0.13` |
| Query frames | every frame (`--keyframe_stride 1`) |
| Primary crop aggregation | mean of all 12 crops; omit `--crop_top_k` |

The 12 headings, FOV, image dimensions, curvature, and refraction values are
the code defaults and are written into the database manifest. Do not store
dense rendered views; `render_db` intentionally keeps descriptors and a few
sample renders only.

Known fixed observer heights above the local surface are:

- Mount Washington: 1.7 m
- Charles River: 1.5 m
- Boston Harbor: 4.0 m

Pohang, Flevoland, and Franconia still need an explicit platform-height
declaration before their expensive database renders. Do not silently accept
the CLI's 1.7 m default for them.

### Portland flight altitude

The three flight profiles are materially different. Their raw
`frames_gps.csv:altitude_m` values are:

| Leg | Raw altitude range / mean | Shape over time | Initial fixed AGL estimate |
|---|---|---|---:|
| leg1 | 6.5–317.2 m / 244.3 m | climb, mostly 230–305 m, descend to 55.6 m | about 175 m |
| leg2 | 41.8–513.1 m / 406.1 m | starts at 227 m, mostly 405–490 m, descends to 41.8 m | about 375 m |
| leg3 | −29.1–409.3 m / 264.6 m | low departure, climb near 390 m, then mostly 240–300 m, descend to 30.7 m | about 270 m |

Do not use one average across all three legs, and do not pass the raw altitude
mean as `observer_height_m`: the renderer expects height **above local
terrain**, not absolute telemetry altitude. The simplest first baseline is one
database per leg using a robust per-leg mean/median AGL recomputed by sampling
the final Maine DEM at each GPS point; the estimates above are starting checks,
not frozen values. This approximates the climb and descent. Per-frame altitude
or an altitude-banked database is not implemented; add it only if a small
height-sensitivity run shows the fixed-per-leg model is inadequate.

## Input contract

For each scoring run, the code consumes:

1. A frozen dataset directory with `pipeline_metadata.json`, `frames_gps.csv`,
   and `panorama/*.jpg`.
2. That dataset's final catalog manifest for the support declaration and
   provenance binding.
3. A primary `HeightField` base path (`surface.npz` + `surface.json`) and zero
   or more fine-to-coarse background base paths.
4. `models/crosslocate/AlpsPhotosToDepthCompact_31_2/converted_weights.npz`.
5. A database directory containing `descriptors.npz` and `manifest.json`.

A typical new raster surface is built with:

```bash
bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:build_surface -- \
  --tiles /path/to/downloaded/*.tif \
  --dst_crs EPSG:26919 \
  --resolution_m 10 \
  --bounds_xy X_MIN_WITH_HALO Y_MIN_WITH_HALO X_MAX_WITH_HALO Y_MAX_WITH_HALO \
  --surface_kind bare_earth_dem \
  --note 'provider, product, acquisition date, horizontal and vertical datum' \
  --output "$FARFIELD_ROOT/artifacts/dem_surfaces/REGION/VERSION/surface"
```

Build one database for each distinct region-and-observer-height combination:

```bash
bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:render_db -- \
  --height_field "$FARFIELD_ROOT/artifacts/dem_surfaces/REGION/VERSION/surface" \
  --background "$FARFIELD_ROOT/artifacts/dem_surfaces/REGION/BACKGROUND/surface" \
  --weights "$FARFIELD_ROOT/models/crosslocate/AlpsPhotosToDepthCompact_31_2/converted_weights.npz" \
  --spacing_m 100 \
  --bounds_xy X_MIN_CANDIDATE Y_MIN_CANDIDATE X_MAX_CANDIDATE Y_MAX_CANDIDATE \
  --max_range_m 30000 \
  --observer_height_m HEIGHT_ABOVE_LOCAL_SURFACE \
  --sky_fill_m -1 \
  --output_dir "$FARFIELD_ROOT/artifacts/depth_render_db/REGION/VERSION"
```

Omit `--background` only when the primary surface really has complete 30 km
coverage. These renders are long-running jobs; let them run to completion.

Score each dataset separately:

```bash
bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:score_localization_inputs -- \
  --dataset DATASET \
  --farfield_root "$FARFIELD_ROOT" \
  --db_dir "$FARFIELD_ROOT/artifacts/depth_render_db/REGION/VERSION" \
  --weights "$FARFIELD_ROOT/models/crosslocate/AlpsPhotosToDepthCompact_31_2/converted_weights.npz" \
  --keyframe_stride 1 \
  --output_dir "$FARFIELD_ROOT/artifacts/retrieval_observations/DATASET/VARIANT"
```

The primary run omits `--crop_top_k`. If the owner requests the robust
aggregation ablation, run a second, separately named output with
`--crop_top_k 6`; this choice is baked into the fields and cannot be changed by
the filter later.

## Output contract

The **required return value for each dataset and scoring variant** is exactly:

```text
artifacts/retrieval_observations/<dataset>/<variant>/
  retrieval_meta.json
  retrieval_fields.npz
```

`retrieval_fields.npz` contains:

- `lat_deg (L,)` and `lon_deg (L,)`: anchor-free lattice node positions
- `scores (K,L,N)`: float16 joint location/heading scores
- `keyframe_idx (K,)`: indices in frozen dataset order
- `pano_ids (K,)`: panorama identities

`retrieval_meta.json` records the dataset, counts, 100 m quantization floor,
database manifest SHA-256, and scorer/weights identity. At 100 m, expect roughly
32 GB uncompressed (about 25 GB compressed, data-dependent) across all 13
primary score artifacts.

Also return the following small audit material:

- every new `surface.json`;
- every reference database `manifest.json`;
- the exact render and score commands/logs;
- a note naming the downloaded elevation product/tile IDs and license; and
- a SHA-256 inventory of the returned files.

The large surfaces and `descriptors.npz` databases are useful for future
rescoring but are **not required to run the filter**. Keep them on the render
machine unless storage has been arranged.

The downstream filter consumes the two retrieval files plus a separately
built current `localization_inputs` artifact:

```bash
bazel run //experimental/overhead_matching/swag/farfield/localization:run_retrieval -- \
  --input_dir /path/to/current/localization_inputs \
  --retrieval_dir /path/to/retrieval_observations/DATASET/VARIANT \
  --run_dir /path/to/run \
  --init uniform \
  --prior_region catalog \
  --margin_m 0 \
  --n_particles N \
  --retrieval_temperature TAU \
  --retrieval_epsilon EPSILON \
  --position_roughening_m METERS \
  --heading_roughening_deg DEGREES
```

Temperature, epsilon, particle count, injection cadence, and roughening are
downstream filter calibration choices. They do not need to be selected before
the score fields are produced.

## Preflight acceptance checklist

Before launching all sequences, complete one short end-to-end sequence and
check:

- database support comes from the copied final catalog manifest and its hash
  is recorded;
- the candidate lattice excludes the halo and has zero dropped no-data nodes;
- every database view has full terrain coverage through 30 km;
- the database manifest records the fixed settings above and finite,
  unit-normalized descriptors;
- returned keyframe count and ordered `pano_ids` exactly match the dataset;
- all score values are finite; and
- `retrieval_meta.json:db_manifest_sha256` matches the returned database
  manifest.

Only then fan out the remaining datasets.
