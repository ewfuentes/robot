load("@pip//:requirements.bzl", "requirement")

# Mapillary far-field sequence discovery and QC. The Graph API token is a
# secret and deliberately absent from this package: MapillaryClient reads
# $MLY_TOKEN, then ~/.config/mapillary/token. Tile blobs cache to
# ~/.cache/mapillary_tiles ($MLY_TILE_CACHE to override).
# Collection (seed -> stitched dataset) still lives in ~/scratch/mappilary;
# see docs/farfield-site-search.md and docs/mapillary-dataset-creation.md.

py_library(
  name="models",
  srcs=["models.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
)

py_library(
  name="api",
  srcs=["api.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
  deps=[
    ":models",
    requirement("requests"),
  ],
)

py_library(
  name="vector_tiles",
  srcs=["vector_tiles.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
  deps=[
    ":api",
    requirement("requests"),
  ],
)

py_library(
  name="farfield_regions",
  srcs=["farfield_regions.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
)

py_library(
  name="discover_tracks_lib",
  srcs=["discover_tracks.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
  deps=[
    ":farfield_regions",
    ":vector_tiles",
  ],
)

py_binary(
  name="discover_tracks",
  srcs=["discover_tracks.py"],
  deps=[":discover_tracks_lib"],
)

py_library(
  name="qc_candidates_lib",
  srcs=["qc_candidates.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
  deps=[":api"],
)

py_binary(
  name="qc_candidates",
  srcs=["qc_candidates.py"],
  deps=[":qc_candidates_lib"],
)

# ---- collection: seed -> dataset (stages 1-3 + orchestrator) ----

py_library(
  name="tiling",
  srcs=["tiling.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
  deps=[":models"],
)

py_library(
  name="geocode",
  srcs=["geocode.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
  deps=[
    ":models",
    requirement("requests"),
  ],
)

py_library(
  name="sequences",
  srcs=["sequences.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
  deps=[
    ":api",
    ":geocode",
    ":models",
    ":tiling",
  ],
)

py_library(
  name="farfield_trajectories",
  srcs=["farfield_trajectories.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
)

py_binary(
  name="seed_to_trajectory",
  srcs=["seed_to_trajectory.py"],
  deps=[
    ":api",
    ":models",
    ":sequences",
    ":tiling",
  ],
)

py_binary(
  name="extract_stitch",
  srcs=["extract_stitch.py"],
  deps=[
    ":api",
    ":models",
  ],
)

py_binary(
  name="mapillary_to_vigor",
  srcs=["mapillary_to_vigor.py"],
  deps=[
    requirement("numpy"),
    requirement("opencv-python"),
  ],
)

py_binary(
  name="run_farfield_collection",
  srcs=["run_farfield_collection.py"],
  deps=[":farfield_trajectories"],
)

py_test(
  name="discover_tracks_test",
  srcs=["discover_tracks_test.py"],
  deps=[":discover_tracks_lib"],
)

py_test(
  name="vector_tiles_test",
  srcs=["vector_tiles_test.py"],
  deps=[":vector_tiles"],
)
