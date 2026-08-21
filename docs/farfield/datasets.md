# Datasets

## The contract

A dataset directory is a **frozen problem definition**:

```
datasets/<name>/
  panorama/f####,<lat>,<lon>,.jpg    equirect frames, GPS in the filename
  frames_gps.csv                     idx, latitude, longitude, dist_m, video_t_s
  intrinsics.csv                     heading_deg + heading_reference + hfov
  extraction_log.csv                 source provenance (Mapillary ids, sequences)
  pano_id_mapping.csv
  pipeline_metadata.json             conventions, video pointer, mount offset
```

`farfield/dataset.py` is the only reader of this contract, and
`farfield:audit_dataset` is its enforcement tool — run it on every new or
modified dataset. It checks the things that fail silently: filename parsing,
frame ordering, table agreement, convention metadata (`north_aligned`, the
azimuth formula, mount-offset qualifiers), GPS plausibility, image integrity,
and whether `video_t_s` actually addresses the frames it claims (the NCC
check that would have caught the charles_river trim incident).

No stage ever writes into a dataset. The one exception is
`dataset_tools:publish_mount_offset`, the explicit, guarded publisher of a
calibrated mount offset (accuracy-validated guard + checksum regeneration).

## Getting datasets

**Self-collected (video + GPS rig):** `dataset_tools:ingest_selfcollect`
turns a frame dump + GPS log into the contract, writing the convention
metadata (including the mount-offset frame qualifiers) as it goes.

**Mapillary:** the `collection/` package: discover candidate tracks in a
region, QC the seeds, resolve a seed into a whole trip, download/stitch, and
convert to the contract (`collection:run_farfield_collection` orchestrates;
each stage is also a standalone tool). The converter records
`azimuth_convention` with the left-edge caveat and the mount-offset frame
note — see `conventions.md` §3.

**Triage tools** (`dataset_tools/`): trajectory timelapse, vehicle-anchor
detection, recording-seam annotation, frame trimming (which rebases seams
and invalidates stale calibrations), and the status table generator.

## Catalogs

Map-side landmark tables (OSM + ENC) are **derived products** and live in
`artifacts/catalogs/<dataset>/<stem>.feather` — never inside `datasets/`.
They are built by `dataset_tools` (Overpass/PBF extraction, ENC extraction,
merge) and trimmed by `dataset_tools:trim_catalog`, whose recall guard is
mandatory: a trim that drops a landmark the pairing labels say we observed
is refused. Every trim carries a provenance sidecar with the exact arguments
and a reproduce command; a trim byte-identical to an existing sibling is
refused (a new version must contain new content).

All feather reading goes through `catalog/schema.py` — both the current
dict-tags layout and the legacy wide layout. The far-field tag vocabulary
(`catalog.catalog.keeps_tag_key`) is the single source the trim and the
loader share.

## The data root

```
<root>/                        (default /data/farfield_matching; $FARFIELD_ROOT overrides)
  datasets/<name>/             frozen problem definitions
  artifacts/<kind>/<dataset>/<version>/   derived products + manifest.json
  artifacts/catalogs/<dataset>/<stem>.feather
  runs/<experiment>/<run>/     localization experiments (+ experiment.md)
  models/<family>/<file>      weights (+ SOURCE.md)
  raw_material/                source material
```

Artifact versions are explicit everywhere — there is no default version or
default catalog anywhere in the tree, because a version default is how a
stage silently reads one version's data against another's.
