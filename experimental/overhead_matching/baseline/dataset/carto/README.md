# OSM Carto tiles (`<Env>/satellite_osm_carto/`)

Labelled OSM Carto (osm.org "Standard") rasters on the exact satellite grid of a
VIGOR-format environment, one PNG per `satellite/` tile with the same stem
(`satellite_<lat>_<lon>.png`). They stand in for the OSM reference tiles that
CVG-Text / CrossText2Loc were trained and evaluated on: their released tiles are
z19 osm.org Carto tiles, 256 px centre-cropped and upscaled x2 to 512 px, i.e.
z20 pixel density, which is what our satellite grid already has (see the
2026-09-25 characterisation under `/data/overhead_matching/scratch/ct2l_osm_tiles/`).

`satellite_osm/` (pymgl, OSM Bright, no labels) is NOT a substitute: the CT2L OSM
checkpoint expects Carto colours, road casings and street/POI labels.

## What is rendered

* Style: openstreetmap-carto **v5.4.0** as shipped in
  `overv/openstreetmap-tile-server:latest` (mapnik 3.1, Noto fonts, its own
  compiled `mapnik.xml`), rendered with python-mapnik inside that container.
* Data: one PostGIS docker volume per **dated Geofabrik extract**, the same extract
  each environment's landmark tables / `satellite_osm/` used
  (`city_pbf_map.py`; the landmark version suffix encodes the date, e.g.
  `v4_202001`, `*_250101`). Newer data is deliberately not used.
* Footprint: `tile_geometry.center_zoom_to_bbox(lat, lon, zoom=20, tile_px)`
  converted to EPSG:3857 metres, `tile_px` = 640 or `satellite_bbox.json:source_px`
  for resolution-normalised environments, exactly as `render_osm_tiles.py`.
* Scale: Carto rules evaluated at **z19** with mapnik
  `scale_factor = px / (bbox_m / 0.2986)` and a 128 px label buffer. At 640 px
  that is scale factor 2 = "z19 tile upscaled x2" = CVG-Text's look.
* **Resolution: HALF** (`px = round(tile_px / 2)`: 320 for a 640 grid, 395 / 391 /
  266 for netherlands_norr / netherlands_veluwe / post_hurricane_ian_sw), i.e. a
  native z19 raster (scale factor 1.0). Consumers upscale x2 (bicubic) at load.
  Measured on the CVG-Text samples: render-at-half then bicubic x2 matches their
  tiles better (NCC 0.616) than rendering at scale factor 2 (0.575), and it is a
  quarter of the bytes: ~10 KB per RGBA PNG, ~17 GB for all 1.67 M tiles.

## Commands

All paths are absolute; run from the repo root. Docker only, no sudo. Output
PNGs come out owned by uid 1000 (`renderer` in the image == `ekf` here).

```bash
C=experimental/overhead_matching/baseline/dataset/carto
D=/data/overhead_matching/datasets/osm_dumps

# 1. Import a dated extract once (idempotent; volume ~5 GB + 30x the PBF size under /var/lib/docker).
#    Refuses to start with < 60 GB + that estimate free on /.
$C/import_region.sh $D/washington-200101.osm.pbf          # -> volume ct2l-osm-db-washington_200101

# 2. Job list for one environment (skips tiles already in satellite_osm_carto/).
bazel run //$C:make_carto_jobs -- --env Seattle --out /tmp/jobs/Seattle.json
#    prints the render command:
$C/run_carto_render.sh washington_200101 /tmp/jobs/Seattle.json \
    /data/overhead_matching/datasets/VIGOR/Seattle/satellite_osm_carto 16

# 3. Verify + contact sheet of 4 random [satellite | carto x2] pairs.
bazel run //$C:check_carto -- --env Seattle \
    --composite /data/overhead_matching/scratch/ct2l_osm_tiles/composites/Seattle_carto_check.png
```

`run_carto_render.sh` keeps one container per region, `ct2l-osm-run-<region>`
(volume + `VIGOR_ROOT` mounted rw at `/vigor`), starts it if stopped, recreates
it if the mount differs, copies `render_carto.py` and the jobs file in, and runs
N `docker exec` shards. Shards skip existing files and write-then-rename, so a
killed run is resumed by re-running steps 2-3. `nightdrive` shares Boston's grid:
`nightdrive/satellite_osm_carto -> ../Boston/satellite_osm_carto` is a symlink.

Region name = PBF stem with `-` -> `_`. Environments and extracts:

| Environment(s)                 | PBF                          | volume                          |
|--------------------------------|------------------------------|---------------------------------|
| Boston, nightdrive, Framingham | massachusetts-260101.osm.pbf | ct2l-osm-db-massachusetts_260101|
| Middletown                     | connecticut-250101.osm.pbf   | ct2l-osm-db-connecticut_250101  |
| NewYork                        | new-york-200101.osm.pbf      | ct2l-osm-db-new_york_200101     |
| Seattle                        | washington-200101.osm.pbf    | ct2l-osm-db-washington_200101   |
| Chicago                        | illinois-200101.osm.pbf      | ct2l-osm-db-illinois_200101     |
| SanFrancisco_mapillary         | norcal-220101.osm.pbf        | ct2l-osm-db-norcal_220101       |
| post_hurricane_ian_sw          | florida-220101.osm.pbf       | ct2l-osm-db-florida_220101      |
| netherlands_norr, _veluwe      | netherlands-250101.osm.pbf   | ct2l-osm-db-netherlands_250101  |

Dated extracts come from `https://download.geofabrik.de/<continent>/<region>-<yymmdd>.osm.pbf`
(e.g. `north-america/us/california/norcal-220101.osm.pbf`).

## Throughput and disk (2026-09-28, 32-core host, 16 shards)

THROUGHPUT_TABLE

## Fidelity gaps vs CVG-Text's tiles

* **Data date.** Theirs is ~2024 osm.org; ours is the dated extract the landmark
  tables use (NY/Chicago/Seattle Jan 2020: fewer POIs, sidewalks, buildings).
* **Carto v5.4.0 vs current osm.org** (v5.8+): < 0.25 % of pixels differ on the
  characterisation samples (colour tweaks, a few icons).
* **Label placement** is per-render, not per-metatile: a label osm.org drops
  because a neighbouring tile claimed it may appear here and vice versa. Font
  hinting differs slightly.
* **No tile-aligned jitter.** Their crops are z19-tile aligned (query point up to
  +-33.6 Mercator m off centre); ours are centred on the satellite tile.
* **Boston footprint.** `Boston/satellite/tile_metadata.csv` says 72.95 m tiles,
  3.3 % wider than the 640 px z20 model the loader, `satellite_osm/` and this
  renderer all assume (1.6 m at each edge). Framingham/Middletown agree with the
  model.
