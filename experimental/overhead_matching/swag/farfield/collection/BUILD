load("@pip//:requirements.bzl", "requirement")

# Mapillary far-field collection: discovery, QC, seed->trajectory stitching,
# download, dataset conversion, and the end-to-end orchestrator (PR 13 of the
# farfield reorg; ported from swag/mapillary_tools + scripts/pbf_coverage).
#
# The Graph API token is a secret and deliberately absent from this package:
# MapillaryClient reads $MLY_TOKEN, then ~/.config/mapillary/token. Tile blobs
# cache to ~/.cache/mapillary_tiles ($MLY_TILE_CACHE to override). The OSM
# extract/.poly caches and the ENC root have no defaults — the orchestrator
# requires --osm_cache_dir / --enc_root.

py_library(
    name = "models",
    srcs = ["models.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    deps = [
        ":models",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "api",
    srcs = ["api.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("requests"),
        ":models",
    ],
)

py_library(
    name = "tiling",
    srcs = ["tiling.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":api",
        ":models",
    ],
)

py_library(
    name = "geocode",
    srcs = ["geocode.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("requests"),
        ":models",
    ],
)

py_library(
    name = "vector_tiles",
    srcs = ["vector_tiles.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("requests"),
        ":api",
    ],
)

py_test(
    name = "vector_tiles_test",
    srcs = ["vector_tiles_test.py"],
    deps = [":vector_tiles"],
)

py_library(
    name = "farfield_regions",
    srcs = ["farfield_regions.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_library(
    name = "farfield_trajectories",
    srcs = ["farfield_trajectories.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_test(
    name = "farfield_trajectories_test",
    srcs = ["farfield_trajectories_test.py"],
    deps = [":farfield_trajectories"],
)

py_library(
    name = "provenance_util",
    srcs = ["provenance_util.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_library(
    name = "geometry_helpers",
    srcs = ["geometry_helpers.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "geometry_helpers_test",
    srcs = ["geometry_helpers_test.py"],
    deps = [":geometry_helpers"],
)

py_library(
    name = "discover_tracks_lib",
    srcs = ["discover_tracks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":farfield_regions",
        ":provenance_util",
        ":vector_tiles",
    ],
)

py_binary(
    name = "discover_tracks",
    srcs = ["discover_tracks.py"],
    deps = [":discover_tracks_lib"],
)

py_test(
    name = "discover_tracks_test",
    srcs = ["discover_tracks_test.py"],
    deps = [":discover_tracks_lib"],
)

py_library(
    name = "qc_candidates_lib",
    srcs = ["qc_candidates.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":api",
        ":provenance_util",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_binary(
    name = "qc_candidates",
    srcs = ["qc_candidates.py"],
    deps = [":qc_candidates_lib"],
)

py_library(
    name = "seed_to_trajectory_lib",
    srcs = ["seed_to_trajectory.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":api",
        ":models",
        ":provenance_util",
        ":tiling",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_binary(
    name = "seed_to_trajectory",
    srcs = ["seed_to_trajectory.py"],
    deps = [":seed_to_trajectory_lib"],
)

py_library(
    name = "extract_stitch_lib",
    srcs = ["extract_stitch.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pillow"),
        ":api",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_binary(
    name = "extract_stitch",
    srcs = ["extract_stitch.py"],
    deps = [":extract_stitch_lib"],
)

py_library(
    name = "mapillary_to_vigor_lib",
    srcs = ["mapillary_to_vigor.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        # matplotlib is a declared dep so --visualize works under bazel (the
        # old target omitted it and crashed on the lazy import).
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("opencv-python"),
        requirement("tqdm"),
        ":models",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_binary(
    name = "mapillary_to_vigor",
    srcs = ["mapillary_to_vigor.py"],
    deps = [":mapillary_to_vigor_lib"],
)

py_test(
    name = "mapillary_to_vigor_test",
    srcs = ["mapillary_to_vigor_test.py"],
    deps = [
        ":mapillary_to_vigor_lib",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "pbf_coverage_lib",
    srcs = ["pbf_coverage.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("shapely"),
    ],
)

py_binary(
    name = "pbf_coverage",
    srcs = ["pbf_coverage.py"],
    deps = [":pbf_coverage_lib"],
)

py_binary(
    name = "run_farfield_collection",
    srcs = ["run_farfield_collection.py"],
    deps = [
        ":extract_stitch_lib",
        ":farfield_trajectories",
        ":geometry_helpers",
        ":mapillary_to_vigor_lib",
        ":pbf_coverage_lib",
        ":seed_to_trajectory_lib",
        "//experimental/overhead_matching/swag/farfield:audit_dataset_lib",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)
