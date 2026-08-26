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

py_library(
    name = "source_publication",
    srcs = ["source_publication.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        FARFIELD + ":artifact",
        FARFIELD + "/catalog:schema",
    ],
)

py_test(
    name = "feather_utils_test",
    srcs = ["feather_utils_test.py"],
    deps = [
        requirement("shapely"),
        ":feather_utils",
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
        ":make_dataset_timelapse_lib",
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
        requirement("pillow"),
        ":make_dataset_timelapse_lib",
        ":trim_dataset_lib",
        FARFIELD + ":artifact",
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

py_library(
    name = "detect_vehicle_anchor_lib",
    srcs = ["detect_vehicle_anchor.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "detect_vehicle_anchor",
    srcs = ["detect_vehicle_anchor.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":detect_vehicle_anchor_lib"],
)

py_test(
    name = "detect_vehicle_anchor_test",
    srcs = ["detect_vehicle_anchor_test.py"],
    deps = [":detect_vehicle_anchor_lib"],
)

py_library(
    name = "make_dataset_timelapse_lib",
    srcs = ["make_dataset_timelapse.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pillow"),
        FARFIELD + ":artifact",
        FARFIELD + ":paths",
        FARFIELD + ":provenance",
        FARFIELD + ":publication",
    ],
)

py_test(
    name = "make_dataset_timelapse_test",
    srcs = ["make_dataset_timelapse_test.py"],
    deps = [
        requirement("pillow"),
        ":make_dataset_timelapse_lib",
        FARFIELD + ":artifact",
    ],
)

py_binary(
    name = "make_dataset_timelapse",
    srcs = ["make_dataset_timelapse.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":make_dataset_timelapse_lib"],
)

py_library(
    name = "dataset_status_table_lib",
    srcs = ["dataset_status_table.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [FARFIELD + ":provenance"],
)

py_binary(
    name = "dataset_status_table",
    srcs = ["dataset_status_table.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":dataset_status_table_lib"],
)

py_test(
    name = "dataset_status_table_test",
    srcs = ["dataset_status_table_test.py"],
    deps = [":dataset_status_table_lib"],
)

py_library(
    name = "backfill_artifact_identity_lib",
    srcs = ["backfill_artifact_identity.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        FARFIELD + ":artifact",
        FARFIELD + ":artifact_identity",
        FARFIELD + ":paths",
        FARFIELD + ":pipeline_lib",
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "backfill_artifact_identity",
    srcs = ["backfill_artifact_identity.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":backfill_artifact_identity_lib"],
)

py_test(
    name = "backfill_artifact_identity_test",
    srcs = ["backfill_artifact_identity_test.py"],
    deps = [
        ":backfill_artifact_identity_lib",
        FARFIELD + ":artifact",
        FARFIELD + ":artifact_identity",
        FARFIELD + ":provenance",
    ],
)

py_library(
    name = "survey_artifact_identity_lib",
    srcs = ["survey_artifact_identity.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        FARFIELD + ":artifact",
        FARFIELD + ":artifact_identity",
        FARFIELD + ":code_provenance",
        FARFIELD + ":paths",
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "survey_artifact_identity",
    srcs = ["survey_artifact_identity.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":survey_artifact_identity_lib"],
)

py_test(
    name = "survey_artifact_identity_test",
    srcs = ["survey_artifact_identity_test.py"],
    deps = [
        ":survey_artifact_identity_lib",
        FARFIELD + ":artifact",
        FARFIELD + ":provenance",
    ],
)

# ---------------------------------------------------------------------------
# Catalog building (OSM + ENC -> feather -> trim)
# ---------------------------------------------------------------------------

py_library(
    name = "extract_landmarks_from_osm_lib",
    srcs = ["extract_landmarks_from_osm.py"],
    data = ["//common/openstreetmap:extract_landmarks_python.so"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("geopandas"),
        requirement("pyarrow"),
        requirement("shapely"),
        ":source_publication",
        FARFIELD + ":artifact",
        FARFIELD + ":provenance",
        FARFIELD + "/catalog:schema",
    ],
)

py_binary(
    name = "extract_landmarks_from_osm",
    srcs = ["extract_landmarks_from_osm.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":extract_landmarks_from_osm_lib"],
)

py_test(
    name = "extract_landmarks_from_osm_test",
    srcs = ["extract_landmarks_from_osm_test.py"],
    data = ["//common/openstreetmap:extract_landmarks_python.so"],
    deps = [
        requirement("shapely"),
        ":extract_landmarks_from_osm_lib",
        ":source_publication",
        FARFIELD + ":artifact",
        FARFIELD + "/catalog:schema",
    ],
)

py_library(
    name = "plot_landmarks_lib",
    srcs = ["plot_landmarks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("shapely"),
        FARFIELD + ":artifact",
        FARFIELD + ":dataset",
        FARFIELD + ":geometry",
        FARFIELD + ":paths",
        FARFIELD + ":provenance",
        FARFIELD + ":publication",
        FARFIELD + "/catalog:schema",
        FARFIELD + "/collection:pbf_coverage_lib",
        FARFIELD + "/viewers:page",
    ],
)

py_binary(
    name = "plot_landmarks",
    srcs = ["plot_landmarks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":plot_landmarks_lib"],
)

py_test(
    name = "plot_landmarks_test",
    srcs = ["plot_landmarks_test.py"],
    deps = [
        requirement("shapely"),
        ":plot_landmarks_lib",
        FARFIELD + ":artifact",
        FARFIELD + ":paths",
        FARFIELD + "/catalog:schema",
    ],
)

py_library(
    name = "download_enc_cells_lib",
    srcs = ["download_enc_cells.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("requests"),
        FARFIELD + ":artifact",
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
        ":download_enc_cells_lib",
        ":feather_utils",
        ":source_publication",
        FARFIELD + ":artifact",
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
        ":download_enc_cells_lib",
        ":extract_landmarks_from_enc_lib",
        ":feather_utils",
        ":source_publication",
        FARFIELD + ":artifact",
        FARFIELD + "/catalog:catalog",
        FARFIELD + "/catalog:schema",
    ],
)

py_library(
    name = "merge_landmark_feathers_lib",
    srcs = ["merge_landmark_feathers.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("geopandas"),
        ":feather_utils",
        ":source_publication",
        FARFIELD + ":artifact",
        FARFIELD + ":provenance",
        FARFIELD + "/catalog:schema",
    ],
)

py_binary(
    name = "merge_landmark_feathers",
    srcs = ["merge_landmark_feathers.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":merge_landmark_feathers_lib"],
)

py_test(
    name = "merge_landmark_feathers_test",
    srcs = ["merge_landmark_feathers_test.py"],
    deps = [
        requirement("shapely"),
        ":merge_landmark_feathers_lib",
        ":source_publication",
        FARFIELD + ":artifact",
        FARFIELD + "/catalog:schema",
    ],
)

py_library(
    name = "landmark_positive_set_lib",
    srcs = ["landmark_positive_set.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        FARFIELD + ":artifact",
        FARFIELD + ":paths",
        FARFIELD + ":provenance",
    ],
)

py_binary(
    name = "landmark_positive_set",
    srcs = ["landmark_positive_set.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":landmark_positive_set_lib"],
)

py_test(
    name = "landmark_positive_set_test",
    srcs = ["landmark_positive_set_test.py"],
    deps = [
        ":landmark_positive_set_lib",
        FARFIELD + ":artifact",
        FARFIELD + ":paths",
    ],
)

py_library(
    name = "trim_catalog_lib",
    srcs = ["trim_catalog.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("geopandas"),
        requirement("numpy"),
        ":landmark_positive_set_lib",
        FARFIELD + ":artifact",
        FARFIELD + ":geometry",
        FARFIELD + ":paths",
        FARFIELD + ":publication",
        FARFIELD + ":provenance",
        FARFIELD + "/catalog:catalog",
        FARFIELD + "/catalog:schema",
        FARFIELD + "/collection:active_catalogs_lib",
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
        ":landmark_positive_set_lib",
        ":trim_catalog_lib",
        FARFIELD + ":artifact",
        FARFIELD + ":paths",
    ],
)
