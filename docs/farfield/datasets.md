# Datasets and catalogs

## The dataset contract

A dataset directory is a **frozen problem definition**:

```
datasets/<name>/
  panorama/f####,<lat>,<lon>,.jpg    equirectangular frames
  frames_gps.csv                     ordered GPS/time records
  intrinsics.csv                     source camera metadata
  extraction_log.csv                 collection provenance
  pano_id_mapping.csv                source-to-canonical frame identity
  pipeline_metadata.json             source video and convention metadata
  nominal_forward.json               approved calibration, when applicable
```

`farfield/dataset.py` is the canonical reader and `farfield:audit_dataset` is
the enforcement tool. The audit catches failures that otherwise look
plausible: filename/table disagreement, frame-order drift, malformed
convention metadata, GPS discontinuity, bad images, and source-video timing
errors. Pipeline stages do not write into this directory.

Self-collected data enters through `dataset_tools:ingest_selfcollect`.
Mapillary data enters through the independently resumable stages under
`collection/`, with `collection:run_farfield_collection` as the orchestrator.
Both paths publish the same camera-frame dataset contract before scientific
processing begins.

## Full catalogs

Map-side OSM/ENC data is derived evidence, not dataset metadata. The active
catalog planner under `collection/active_catalogs.py` is report-first:

1. bind the exact frozen trajectory tables and their hashes;
2. bind caller-selected complete source files and ENC selection records;
3. compute and review the source-coverage plan;
4. publish only when the caller supplies the expected plan digest.

OSM extraction reads the complete caller-pinned PBF and builds the geometry
index required for ways and relations. It does not depend on an external
smart-pre-extraction step. Invalid source geometry is repaired fail-closed;
every repair is deterministic and recorded in an exact diagnostic ledger.
ENC extraction and OSM/ENC merging are similarly provenance-bound.

The result is a typed full-catalog artifact plus a separate typed
`catalog_coverage` review artifact. Coverage is diagnostic evidence and is not
a hidden input to matching.

## Pruned catalogs

`dataset_tools:trim_catalog` applies the shared far-field semantic vocabulary
and any reviewed spatial policy to a typed full catalog. A governed
trajectory-union clip carries a strict, digest-bound plan containing the
canonical trajectory sources, recomputed union, area/buffer policy, resolved
bounds, coordinate system, and intended output dataset. The producer
revalidates the live sources before and during transactional publication.

This is deliberately post-deduplication representative-point clipping: it
reduces matching workload without pretending to be a new source extraction.
The exact policy and values belong in the reviewed plan, not in this document.
All feather access goes through `catalog/schema.py`; semantic pruning uses the
one vocabulary in `catalog/catalog.py`.

## Added sources

A source the stage-5 planner does not attest (today: Overture Places) enters
as a **derived** catalog, never by rebuilding the full catalog:

1. `dataset_tools:extract_landmarks_from_overture` turns a pinned-release
   `overturemaps download --type=place` GeoParquet into a typed source Feather
   under `raw_material/catalog_sources/<dataset>/`. Places are mapped onto the
   OSM tag vocabulary by taxonomy hierarchy; each row keeps its per-record
   source licences in `overture:*` tags that the keep vocabulary prunes at
   load.
2. `dataset_tools:add_catalog_source` appends that Feather to a published
   catalog as a new CATALOGS artifact with exactly one catalog upstream, so
   `catalog/lineage.py` still terminates at the full catalog's coverage
   attestation. A source row whose normalised name matches a catalog row
   within `--dedupe_name_radius_m` is a duplicate and is recorded, not added.
3. `trim_catalog` runs on the result as on any other catalog, so added rows
   face the same far-field rules as OSM rows.

`landmark_type` names the source (`osm`, `enc`, `overture`) and matcher
landmark ids are namespaced by it. See decisions.md, 2026-09-03.

## Data-root layout

```
<root>/
  datasets/<dataset>/                         frozen problem definitions
  artifacts/<kind>/<dataset>/<version>/      immutable typed artifacts
  builds/<dataset>/<build>/                  mutable orchestration records
  runs/<experiment>/<run>/                   completed localization runs
  models/<family>/<checkpoint>               weights and source records
  raw_material/                               retained source material
  archive/                                    explicitly retired evidence
```

`builds/` contains only orchestration state such as `build_config.json`. It
is not a scientific artifact lane. Current
scientific artifact kinds are:

- `pinhole_images`, `frame_landmarks`, and `object_tracks`;
- `semantic_audits`, `bearing_observations`, and `landmark_matches`;
- `alignment_diagnostics` and `localization_inputs`;
- `catalogs`.

Artifact versions are explicit. There is no default catalog, detection set,
or track set: silently resolving “latest” would make a plausible but false
lineage. Old evidence kept for backtesting belongs under an explicit archive
subdirectory rather than beside current versions.
