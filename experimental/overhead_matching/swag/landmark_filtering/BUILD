load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "pipeline_config",
    srcs = ["pipeline_config.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("pyyaml"),
        "//common/python:serialization",
    ],
)

py_library(
    name = "artifact_schema",
    srcs = ["artifact_schema.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        "//common/python:serialization",
        ":pipeline_config",
    ],
)

py_library(
    name = "bearing_geometry",
    srcs = ["bearing_geometry.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        # The camera frame has one definition and it lives in pano_geometry.
        "//experimental/overhead_matching/swag/landmark_filtering/object_tracking:pano_geometry",
        requirement("numpy"),
    ],
)

py_test(
    name = "bearing_geometry_test",
    srcs = ["bearing_geometry_test.py"],
    deps = [
        requirement("numpy"),
        ":bearing_geometry",
        # The test asserts bearing_geometry delegates to the canonical frame.
        "//experimental/overhead_matching/swag/landmark_filtering/object_tracking:pano_geometry",
    ],
)

py_test(
    name = "artifact_schema_test",
    srcs = ["artifact_schema_test.py"],
    deps = [
        ":artifact_schema",
        ":pipeline_config",
    ],
)

py_library(
    name = "ingest",
    srcs = ["ingest.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":artifact_schema",
        ":bearing_geometry",
        ":pipeline_config",
    ],
)

py_test(
    name = "ingest_test",
    srcs = ["ingest_test.py"],
    deps = [
        ":artifact_schema",
        ":ingest",
        ":pipeline_config",
    ],
)

py_library(
    name = "heuristic_filters",
    srcs = ["heuristic_filters.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":artifact_schema",
        ":bearing_geometry",
        ":pipeline_config",
    ],
)

py_test(
    name = "heuristic_filters_test",
    srcs = ["heuristic_filters_test.py"],
    deps = [
        requirement("msgspec"),
        ":artifact_schema",
        ":heuristic_filters",
        ":pipeline_config",
    ],
)

py_library(
    name = "semantic_similarity",
    srcs = ["semantic_similarity.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
        "//experimental/overhead_matching/swag/data:landmark_correspondence_dataset",
        "//experimental/overhead_matching/swag/evaluation:correspondence_matching",
        "//experimental/overhead_matching/swag/model:landmark_correspondence_model",
        ":artifact_schema",
        ":pipeline_config",
    ],
)

py_test(
    name = "semantic_similarity_test",
    srcs = ["semantic_similarity_test.py"],
    deps = [
        requirement("numpy"),
        ":artifact_schema",
        ":semantic_similarity",
    ],
)

py_library(
    name = "tracking",
    srcs = ["tracking.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
        ":artifact_schema",
        ":bearing_geometry",
        ":heuristic_filters",
        ":pipeline_config",
        ":semantic_similarity",
    ],
)

py_test(
    name = "tracking_test",
    srcs = ["tracking_test.py"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        ":artifact_schema",
        ":pipeline_config",
        ":semantic_similarity",
        ":tracking",
    ],
)

py_library(
    name = "triangulation",
    srcs = ["triangulation.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
        ":artifact_schema",
        ":bearing_geometry",
        ":pipeline_config",
    ],
)

py_test(
    name = "triangulation_test",
    srcs = ["triangulation_test.py"],
    deps = [
        requirement("numpy"),
        ":pipeline_config",
        ":triangulation",
    ],
)

py_library(
    name = "yaw_offset",
    srcs = ["yaw_offset.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":artifact_schema",
        ":bearing_geometry",
        ":pipeline_config",
        ":triangulation",
    ],
)

py_test(
    name = "yaw_offset_test",
    srcs = ["yaw_offset_test.py"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        ":artifact_schema",
        ":pipeline_config",
        ":yaw_offset",
    ],
)

py_binary(
    name = "filter_run_viewer",
    srcs = ["filter_run_viewer.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("flask"),
        requirement("geopandas"),
        requirement("pyarrow"),
        ":artifact_schema",
        ":bearing_geometry",
        ":pipeline_config",
    ],
)

py_binary(
    name = "run_filter_pipeline",
    srcs = ["run_filter_pipeline.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":artifact_schema",
        ":heuristic_filters",
        ":ingest",
        ":pipeline_config",
        ":tracking",
        ":triangulation",
        ":yaw_offset",
    ],
)
