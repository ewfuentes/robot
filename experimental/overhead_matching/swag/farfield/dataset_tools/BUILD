load("@pip//:requirements.bzl", "requirement")

FARFIELD = "//experimental/overhead_matching/swag/farfield"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

py_library(
    name = "checksums",
    srcs = ["checksums.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_library(
    name = "feather_utils",
    srcs = ["feather_utils.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("geopandas"),
        requirement("pandas"),
        requirement("shapely"),
        FARFIELD + ":geometry",
        FARFIELD + "/catalog:schema",
    ],
)

# ---------------------------------------------------------------------------
# Dataset ingest, trimming, triage
# ---------------------------------------------------------------------------

py_library(
    name = "ingest_selfcollect_lib",
    srcs = ["ingest_selfcollect.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        FARFIELD + ":geometry",
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "ingest_selfcollect",
    srcs = ["ingest_selfcollect.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":ingest_selfcollect_lib"],
)

py_test(
    name = "ingest_selfcollect_test",
    srcs = ["ingest_selfcollect_test.py"],
    deps = [
        requirement("pillow"),
        ":ingest_selfcollect_lib",
        FARFIELD + ":dataset",
        FARFIELD + ":geometry",
    ],
)

py_library(
    name = "trim_dataset_lib",
    srcs = ["trim_dataset.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":checksums",
        FARFIELD + ":geometry",
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "trim_dataset",
    srcs = ["trim_dataset.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":trim_dataset_lib"],
)

py_test(
    name = "trim_dataset_test",
    srcs = ["trim_dataset_test.py"],
    deps = [
        ":trim_dataset_lib",
        FARFIELD + ":geometry",
    ],
)

py_library(
    name = "annotate_recording_seams_lib",
    srcs = ["annotate_recording_seams.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        FARFIELD + ":geometry",
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "annotate_recording_seams",
    srcs = ["annotate_recording_seams.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":annotate_recording_seams_lib"],
)

py_binary(
    name = "detect_vehicle_anchor",
    srcs = ["detect_vehicle_anchor.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "make_dataset_timelapse",
    srcs = ["make_dataset_timelapse.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pillow"),
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "dataset_status_table",
    srcs = ["dataset_status_table.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        FARFIELD + ":geometry",
        FARFIELD + ":provenance",
    ],
)

# ---------------------------------------------------------------------------
# The one guarded writer of dataset mount-offset metadata
# ---------------------------------------------------------------------------

py_library(
    name = "publish_mount_offset_lib",
    srcs = ["publish_mount_offset.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":checksums",
        FARFIELD + ":dataset",
        FARFIELD + ":geometry",
        FARFIELD + ":paths",
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "publish_mount_offset",
    srcs = ["publish_mount_offset.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":publish_mount_offset_lib"],
)

py_test(
    name = "publish_mount_offset_test",
    srcs = ["publish_mount_offset_test.py"],
    deps = [
        ":checksums",
        ":publish_mount_offset_lib",
        FARFIELD + ":dataset",
        FARFIELD + ":geometry",
        FARFIELD + ":testing",
    ],
)

# ---------------------------------------------------------------------------
# Catalog building (OSM + ENC -> feather -> trim)
# ---------------------------------------------------------------------------

py_library(
    name = "download_enc_cells_lib",
    srcs = ["download_enc_cells.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("requests"),
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "download_enc_cells",
    srcs = ["download_enc_cells.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":download_enc_cells_lib"],
)

py_test(
    name = "download_enc_cells_test",
    srcs = ["download_enc_cells_test.py"],
    deps = [
        ":download_enc_cells_lib",
        FARFIELD + ":provenance",
    ],
)

py_library(
    name = "extract_landmarks_from_enc_lib",
    srcs = ["extract_landmarks_from_enc.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("geopandas"),
        requirement("pandas"),
        requirement("pyogrio"),
        ":feather_utils",
        FARFIELD + ":provenance",
        FARFIELD + "/catalog:schema",
    ],
)

py_binary(
    name = "extract_landmarks_from_enc",
    srcs = ["extract_landmarks_from_enc.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":extract_landmarks_from_enc_lib"],
)

py_test(
    name = "extract_landmarks_from_enc_test",
    srcs = ["extract_landmarks_from_enc_test.py"],
    deps = [
        requirement("geopandas"),
        ":extract_landmarks_from_enc_lib",
        ":feather_utils",
        FARFIELD + "/catalog:catalog",
        FARFIELD + "/catalog:schema",
    ],
)

py_binary(
    name = "merge_landmark_feathers",
    srcs = ["merge_landmark_feathers.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("geopandas"),
        ":feather_utils",
        FARFIELD + ":provenance",
        FARFIELD + "/catalog:schema",
    ],
)

py_library(
    name = "landmark_positive_set_lib",
    srcs = ["landmark_positive_set.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        FARFIELD + ":paths",
        FARFIELD + ":provenance",
        FARFIELD + "/catalog:catalog",
    ],
)

py_binary(
    name = "landmark_positive_set",
    srcs = ["landmark_positive_set.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":landmark_positive_set_lib"],
)

py_library(
    name = "trim_catalog_lib",
    srcs = ["trim_catalog.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("geopandas"),
        requirement("numpy"),
        ":landmark_positive_set_lib",
        FARFIELD + ":geometry",
        FARFIELD + ":paths",
        FARFIELD + ":provenance",
        FARFIELD + "/catalog:catalog",
        FARFIELD + "/catalog:schema",
    ],
)

py_binary(
    name = "trim_catalog",
    srcs = ["trim_catalog.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":trim_catalog_lib"],
)

py_test(
    name = "trim_catalog_test",
    srcs = ["trim_catalog_test.py"],
    deps = [
        requirement("geopandas"),
        requirement("numpy"),
        ":trim_catalog_lib",
        FARFIELD + "/catalog:catalog",
    ],
)

py_library(
    name = "migrate_data_root_lib",
    srcs = ["migrate_data_root.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        FARFIELD + ":paths",
        FARFIELD + ":provenance",
        FARFIELD + "/viewers:indexes",
    ],
)

py_binary(
    name = "migrate_data_root",
    srcs = ["migrate_data_root.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":migrate_data_root_lib"],
)

py_test(
    name = "migrate_data_root_test",
    srcs = ["migrate_data_root_test.py"],
    deps = [
        ":migrate_data_root_lib",
        FARFIELD + ":testing",
    ],
)
