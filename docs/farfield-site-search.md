# Finding far-field sites, instead of stumbling onto them

`docs/mapillary-dataset-creation.md` covers turning a Mapillary app link into a
dataset. This covers the step before it: deciding *which* link to look for, over
regions nobody has browsed.

Every one of the 22 trajectories in `farfield_trajectories.py` began as a URL
someone found by hand, and all 22 are water — ferries, harbours, one river. That
is a biased sample of what the method needs, not a specification of it, and the
bias came from the search being manual.

## What a far-field site actually requires

Water gives an empty foreground. It is one way to get one, and it is not the
best one, because it does nothing for a fourth requirement that matters as much:

1. **An empty foreground**, so nothing occludes the 5–50 km band. Water,
   elevation, aridity and flatness all buy this.
2. **Landmarks that clear the horizon at that range and exist in a database.**
   This is a hard geometric gate, not a preference — see the table below.
3. **Mapillary coverage from a moving platform**, ideally 360, ideally
   vehicle-mounted.
4. **Bearings spread around the circle.** A harbour crossing puts nearly all its
   landmarks in one wedge, which constrains position across the sightlines and
   barely at all along them.

### Requirement 2 is arithmetic

With an observer 2 m up, `farfield_viewshed.horizon_range_km` gives:

| landmark | height | max range |
|---|---|---|
| park bench | 1 m | 9 km |
| house | 10 m | 18 km |
| water tower | 30 m | 26 km |
| church spire | 45 m | 31 km |
| chimney | 80 m | 40 km |
| radio mast | 150 m | 52 km |
| hill | 1000 m | 126 km |
| alpine peak | 3000 m | 215 km |

Anything under ~50 m tall is capped near 30 km however flat the ground. That is
why peaks and masts dominate a far-field catalog and why most of a harbour
feather cannot contribute to one at any range.

### Requirement 4 is the one a landmark count hides

Far and near landmarks do different jobs, and conflating them is the mistake the
scoring is built to avoid:

* A bearing to a landmark at range R has sensitivity `∂θ/∂p = 1/R` to observer
  **position** and sensitivity 1 to **heading**, at any range. A peak at 50 km
  contributes ~10⁴ times less position information than a building at 500 m.
* So the far field's job is heading — which, under design doc §5.2 where
  position routes through the heading state, is what stops a dead-reckoned track
  from rotating. Near landmarks fix where you are along it.

`site_metrics` therefore reports `n_far` and `axial_spread` (the heading story)
separately from `pos_sigma_major_m` / `pos_sigma_minor_m` (the position story).

`axial_spread` uses **doubled-angle** circular variance. Bearing information
enters as an outer product `u uᵀ`, so `u` and `−u` are identical: landmarks dead
ahead and dead astern constrain the same axis and are geometrically redundant.
An ordinary circular variance scores that pair 1.0, maximally spread, which is
exactly backwards. Doubled, it scores 0.

## The pipeline

Four tools, all bazel targets: discovery and QC in
`//experimental/overhead_matching/swag/mapillary_tools`, scoring in
`swag/scripts`. Discovery and QC need the Graph API token; it is a secret and
never lives in the repo — `MapillaryClient` reads `$MLY_TOKEN`, then
`~/.config/mapillary/token` (chmod 600). Tile blobs cache to
`~/.cache/mapillary_tiles` (`$MLY_TILE_CACHE` overrides). Scoring needs no
token.

```bash
MT=//experimental/overhead_matching/swag/mapillary_tools

# 1. what tracks exist? (seconds)
bazel run $MT:discover_tracks -- --list_regions
bazel run $MT:discover_tracks -- --region geneva_lakeshore --output /tmp/tracks.json

# 2. what is there to see? (Overpass, minutes)
bazel run //experimental/overhead_matching/swag/scripts:farfield_landmarks -- \
    --bbox 6.10 46.30 7.00 46.60 --output /tmp/landmarks.json

# 3. what can each track actually see? (~0.3 s per observer position)
bazel run //experimental/overhead_matching/swag/scripts:farfield_viewshed -- \
    --tracks /tmp/tracks.json --landmarks /tmp/landmarks.json \
    --output /tmp/scored.json --n_samples 8

# 4. is the capture usable? GPS consistency + frame density, one metadata
#    fetch per seed, no downloads. Fail = backtracking / oscillating GPS or
#    >20 m between frames; occasional jumps are excluded by design.
#    Thresholds calibrated on collected sets with known verdicts.
bazel run $MT:qc_candidates -- --seeds <pKey>,<pKey> --output /tmp/qc.json
bazel run $MT:qc_candidates -- --local /data/farfield_matching/datasets/seattle

# 5. registry stanzas for the winners
bazel run $MT:discover_tracks -- --region geneva_lakeshore --scored /tmp/scored.json \
    --emit_registry --top 5
```

A single observer position can also be checked directly, which is the fastest
way to sanity-check a hunch:

```bash
bazel run //experimental/overhead_matching/swag/scripts:farfield_viewshed -- \
    --landmarks /tmp/landmarks.json --lat 46.5060 --lon 6.6290 --max_range_km 90
```

## Discovery: vector tiles, not `/images`

`/images` cannot answer "where is there coverage like X". It rejects any bbox
over 0.010 square degrees *and* separately rejects dense areas on result volume,
both as HTTP 500, so a region sweep subdivides exponentially — the collection
README records depth 10 in SF, Seattle and London, never finishing. It is the
right endpoint for the neighbourhood of a point you already have, which is
exactly what stitching uses it for, and the wrong one for search.

The coverage vector tiles have no area cap and no result cap. Lake Geneva is
**66 tiles and 2 seconds** at z12, returning 385 candidate tracks over 3,737 km.

```
tileset mly1_public/2/{z}/{x}/{y}     (verified with probe_layers, not assumed)
  z4-5    overview   point
  z6-14   sequence   linestring, whole sequence
  z14     image      point, individual images
```

The `sequence` layer carries `id`, `is_pano`, `captured_at`, `creator_id`,
`quality_score`, `foot`, and **`image_id`** — which is exactly the seed pKey
`seed_to_trajectory.py` stitches from. A discovered candidate therefore drops
straight into the registry with no URL-copying step.

Three things tiles do *not* give you:

* **Properties are a subset.** No camera model, no per-image compass angle, no
  `camera_parameters`. Anything deciding *how to convert* a capture still needs
  `/images`. Tiles find candidates; the Graph API qualifies them.
* **Geometry is simplified per zoom**, quantised to a 4096-unit grid — ~2.4 m at
  z12. Fine for screening, and stage 1 re-fetches at full resolution anyway.
* **Sequences are clipped at tile edges**, so a long track arrives as one
  feature per tile it crosses. `merge_tile_features` reassembles them in
  nearest-neighbour order from the westernmost endpoint; joining in tile-fetch
  order instead zig-zags a diagonal track and inflates its length.

Remember a tile sequence is one Mapillary *fragment*. Stitching typically grows
it by an order of magnitude — Folkestone went 500 images to 10,711.

## Things that will bite you

**Overpass requires a User-Agent.** `overpass-api.de` rejects `requests`' default
`python-requests/x.y.z` with **HTTP 406 Not Acceptable**, for every query, valid
or not. The identical query with any other UA returns 200. A 406 reads like a
malformed-query error and sends you off debugging Overpass QL for an hour.

**Overload arrives as HTTP 504, and retrying at the same size cannot work.**
`_fetch_cell` splits the cell instead, the same way `mapillary_lib.tiling
.adaptive_subdivide` handles a too-large Mapillary bbox. Density varies by more
than an order of magnitude between an alpine massif and open water, so no fixed
grid is correct in advance.

**Filter building heights server-side.** Over Lausanne, 1,201 buildings carry a
`height` tag and 2 of them clear 60 m. Filtering in Python instead transfers
600× the data, and it is that volume, not the tag lookup, that pushes a cell
into a 504.

**Peaks and masts need opposite height treatment.** A mountain is already in the
DEM, so its height must be *read from* it and its `structure_height_m` must be
zero — trusting the `ele` tag instead double-counts and puts Mont Blanc at
9.6 km. A mast is absent from the DEM, which models bare ground, so its height
must be *added* or it never clears a horizon it physically dominates. Same
catalog, opposite rule; `Landmark.in_dem` selects between them.

**`natural=cliff` is excluded, deliberately.** It is a linear way, so it enters a
point catalog as the centroid of its line — a position the observer sees nothing
at, on a feature with no name to match. It is also not a rounding error: over the
Lake Geneva bbox, cliffs were 18,741 of 31,486 rows, 60% of the catalog, all of
it inflating visibility counts with things no matcher can associate.

**Do not decimate the DEM, and never compare scores across strides.** Measured
over Lake Geneva against native 30 m:

| site | stride 1 | subsample s3 | max-pool s3 |
|---|---|---|---|
| Lausanne | 457 | 644 | 656 |
| Nyon | 569 | 933 | 1114 |

Subsampling (`grid[::3, ::3]`) steps over narrow ridge crests, so things behind
them become spuriously visible. Max-pooling does not fix it either: the pooled
grid serves *both* as the occluding terrain, where the maximum is genuinely
conservative, and as the source of target and observer ground elevations, where
the maximum lifts every landmark on sloped ground. The second effect dominates,
so pooling is *more* optimistic than subsampling. A one-sided guarantee needs
separate grids, which gives back the memory decimation was for. Default is
stride 1.

**Peaks occlude themselves without a near-target exclusion.** The DEM post at a
summit *is* the summit, so if it stays in the occluder set every peak in the
catalog returns grazing ≈ 0 and half of them fail the visibility cut. Terrain
within `exclude_near_target_m` of the target is the landmark's own massif, not an
occluder.

**Refraction is `k = 0.13`, not the 4/3 earth.** The familiar 4/3 figure is
`k = 0.25`, the *radio* convention; using it overstates optical sightlines by
roughly 8% in range, enough to promote sites whose landmarks are below the
horizon.

**Read `grazing_deg`, not `visible`.** A landmark clearing the skyline by 0.02°
is a geometric technicality — at 40 km that is 35 m of clearance, inside SRTM's
own vertical error and well inside what a tree line or a haze layer removes.
`site_metrics` cuts at 0.05° by default. Marginal sightlines are also very
sensitive to exact observer position: Mont Blanc is visible from Geneva's Pâquis
quay at +0.12° and occluded from a point 700 m away, which is a real property of
that view, not an artefact.

## Validation

The viewshed was checked against known Swiss sightlines from the Lake Geneva
lakefront (SRTM, stride 1):

| check | result |
|---|---|
| DEM at Lausanne shore | 374 m (truth 372 m) |
| DEM at Mont Blanc summit | 4786 m (truth 4808 m — SRTM under-reads sharp summits) |
| La Dôle from Lausanne | 41.6 km, bearing 258°, visible |
| Dent d'Oche from Lausanne | 18.7 km, bearing 155°, visible |
| Mont Blanc from Geneva Pâquis quay | 69.5 km, **visible** (+0.12°) |
| Mont Blanc from Lausanne waterfront | 77.0 km, **occluded** (−0.46°) |
| Mont Blanc from Sauvabelin tower (673 m) | 80.5 km, **visible** (+0.03°) |

The last three are the discriminating case: the iconic Mont Blanc view is from
Geneva, not from the Lausanne waterfront, and in Lausanne you need elevation to
get it. The model reproduces that.

**Look up landmarks by position, not by name.** Mont Blanc is
`Mont Blanc / Monte Bianco` in OSM, and the Dents du Midi's summit node is
`Haute Cime`. A name query says "not in catalog" for four major peaks that are
all present.

## Region registry

`farfield_regions.py` lists candidate areas with a `geometry` label, so a search
that comes back empty says something about the *category* rather than the place.
Current coverage from one discovery pass each (tracks ≥ 3 km, non-foot):

| region | geometry | tracks | pano | total km | pano km |
|---|---|---|---|---|---|
| dubai_desert | supertall | 4376 | 307 | 32467 | 2801 |
| rainier_puget | isolated_peak | 3072 | 299 | 22396 | 2063 |
| oresund_bridge | causeway | 2883 | 22 | 18774 | 116 |
| flevoland_polder | flat_plain | 2540 | 258 | 20026 | 2646 |
| fuji_kanto | isolated_peak | 1375 | 91 | 10159 | 1162 |
| salt_lake_valley | intermontane_basin | 1030 | 339 | 7753 | 1809 |
| vancouver_georgia | coastal | 693 | 22 | 5646 | 200 |
| kansas_plains | flat_plain | 488 | 3 | 4701 | 62 |
| geneva_lakeshore | alpine_lake | 385 | 44 | 3737 | 438 |

For scale: the entire existing registry is **431 pano-km** across 22
trajectories. Salt Lake Valley alone offers 1,809.

Kansas is the informative negative — 488 tracks but only 3 panoramas, so the US
plains category is real geometrically and thin on 360 coverage. Øresund is
similar for a different reason: 2,883 sequences, almost all perspective dashcam.

## Related

* `docs/mapillary-dataset-creation.md` — what to do with a seed once you have one
* `swag/mapillary_tools/discover_tracks.py` — region → candidate tracks,
  grouped into probable single outings (same creator + endpoints ≤ 3 km +
  start times ≤ 2 h; mass-duplicated timestamps = broken clock, never merge)
* `swag/mapillary_tools/qc_candidates.py` — GPS-consistency + density QC per seed
* `swag/mapillary_tools/farfield_regions.py` — the candidate region registry
* `swag/mapillary_tools/vector_tiles.py` — MVT decoder + tile client
* tests: `bazel test //experimental/overhead_matching/swag/mapillary_tools/...`
  (the live API check in vector_tiles_test still gates on `MLY_LIVE=1`)
