# OSM-tile baseline dataset

Renders rasterized OSM tiles into VIGOR's `<City>/satellite_osm/`, drop-in
parallel to `<City>/satellite/`. The existing `vigor_dataset.py` loader picks
them up by setting `VigorDatasetConfig.satellite_subdir = "satellite_osm"`.

This implements the rasterized-OSM data side of the Zhou et al. (IROS 2021)
baseline so we can train `swag/model/patch_embedding.py` on rendered map
tiles instead of overhead imagery, against the same panorama side.

## Pipeline

1. **planetiler** (Java jar, fetched via `@planetiler_jar` http_file) reads a
   state-level `.osm.pbf` from `/data/overhead_matching/datasets/osm_dumps/`
   and writes a vector `.mbtiles` file in OpenMapTiles schema, cropped to the
   VIGOR city bbox + a 1 km buffer.
2. **OSM Bright GL style** (`@osm_bright_gl_style`) + **Noto Sans glyphs**
   (`@openmaptiles_fonts`) define how the vector tiles get rasterized.
3. **pymgl** (Python wrapper around maplibre-native) opens the MBTiles via
   SQLite, applies the patched style, and rasterizes 640×640 PNGs at the same
   web-mercator bbox as each existing satellite tile.

## Smoke test (Seattle)

```bash
bazel run //experimental/overhead_matching/baseline/dataset:bake_mbtiles -- \
    --city Seattle \
    --vigor-root /data/overhead_matching/datasets/VIGOR
bazel run //experimental/overhead_matching/baseline/dataset:render_osm_tiles -- \
    --city Seattle --limit 1 \
    --vigor-root /data/overhead_matching/datasets/VIGOR
```

The first command writes `/data/overhead_matching/baseline/mbtiles/washington.mbtiles`
(~3 MB, ~1 min on first run with auxiliary downloads, ~30 s thereafter).
The second writes one PNG into `/data/overhead_matching/datasets/VIGOR/Seattle/satellite_osm/`.

## Bake every city

```bash
./experimental/overhead_matching/baseline/dataset/scripts/bake_all_cities.sh
```

The script skips already-baked MBTiles and already-rendered tiles, so it's safe
to re-run after a partial run. Approximate timings on a typical workstation:

| City         | tiles  | bake    | render |
|--------------|--------|---------|--------|
| Seattle      | 20,776 | ~1 min  | ~3 min |
| SanFrancisco | 24,255 | ~2 min  | ~5 min |
| NewYork      | 23,279 | ~1 min  | ~5 min |
| Chicago      | 44,616 | ~1 min  | ~8 min |
| Boston       | 44,290 | ~1 min  | ~8 min |

## Loading the rendered tiles

```python
from experimental.overhead_matching.swag.data.vigor_dataset import (
    VigorDataset, VigorDatasetConfig,
)
ds = VigorDataset(
    Path("/data/overhead_matching/datasets/VIGOR/Seattle"),
    VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        satellite_subdir="satellite_osm",   # NEW
    ),
)
```

## Known limitations

- The v1.11 release zip of osm-bright-gl-style does not bundle a sprite sheet
  (it lives on github-pages). `style_template.render_style()` removes the
  sprite reference; layers using `icon-image` quietly skip rendering. This
  affects POI markers but not roads, water, buildings, or land cover, which
  are the dominant signal for the CNN.
