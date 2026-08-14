load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "pano_geometry",
    srcs = ["pano_geometry.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
    ],
)

py_test(
    name = "pano_geometry_test",
    srcs = ["pano_geometry_test.py"],
    deps = [
        requirement("numpy"),
        ":pano_geometry",
    ],
)

py_library(
    name = "heading",
    srcs = ["heading.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
    ],
)

py_test(
    name = "heading_test",
    srcs = ["heading_test.py"],
    deps = [
        requirement("numpy"),
        ":heading",
    ],
)

py_library(
    name = "video_frames",
    srcs = ["video_frames.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python"),
    ],
)

py_library(
    name = "viz_common",
    srcs = ["viz_common.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pillow"),
        ":pano_geometry",
    ],
)

py_binary(
    name = "m1_heading_windows",
    srcs = ["m1_heading_windows.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pillow"),
        ":heading",
        ":pano_geometry",
        ":video_frames",
        ":viz_common",
        "//experimental/overhead_matching/swag/landmark_filtering:ingest",
        "//experimental/overhead_matching/swag/landmark_filtering:pipeline_config",
    ],
)

py_library(
    name = "track_builder",
    srcs = ["track_builder.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
        ":pano_geometry",
    ],
)

py_test(
    name = "track_builder_test",
    srcs = ["track_builder_test.py"],
    deps = [
        requirement("numpy"),
        ":track_builder",
    ],
)

py_library(
    name = "sam_backend",
    srcs = ["sam_backend.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        requirement("sam-2"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
    ],
)

py_library(
    name = "range_runner",
    srcs = ["range_runner.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":heading",
        ":pano_geometry",
        ":sam_backend",
        ":track_builder",
        ":video_frames",
        ":viz_common",
        "//experimental/overhead_matching/swag/landmark_filtering:ingest",
        "//experimental/overhead_matching/swag/landmark_filtering:pipeline_config",
    ],
)

py_binary(
    name = "m3_build_tracks",
    srcs = ["m3_build_tracks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":range_runner",
        ":track_builder",
        ":viz_common",
    ],
)

py_binary(
    name = "m4_level1_review",
    srcs = ["m4_level1_review.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":pano_geometry",
        ":track_builder",
        ":viz_common",
        "//experimental/overhead_matching/swag/landmark_filtering:ingest",
        "//experimental/overhead_matching/swag/landmark_filtering:pipeline_config",
    ],
)

py_library(
    name = "semantic_audit",
    srcs = ["semantic_audit.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        requirement("pydantic"),
        ":pano_geometry",
        ":track_builder",
    ],
)

py_test(
    name = "semantic_audit_test",
    srcs = ["semantic_audit_test.py"],
    deps = [
        ":semantic_audit",
    ],
)

py_binary(
    name = "m5_build_audit_requests",
    srcs = ["m5_build_audit_requests.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":semantic_audit",
        "//experimental/overhead_matching/swag/landmark_filtering:ingest",
        "//experimental/overhead_matching/swag/landmark_filtering:pipeline_config",
        "//experimental/overhead_matching/swag/scripts:vertex_batch_manager_lib",
    ],
)

py_binary(
    name = "m5_audit_results_viewer",
    srcs = ["m5_audit_results_viewer.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":semantic_audit",
        "//experimental/overhead_matching/swag/landmark_filtering:ingest",
        "//experimental/overhead_matching/swag/landmark_filtering:pipeline_config",
    ],
)

py_library(
    name = "track_merge",
    srcs = ["track_merge.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":pano_geometry"],
)

py_test(
    name = "track_merge_test",
    srcs = ["track_merge_test.py"],
    deps = [
        ":pano_geometry",
        ":track_merge",
    ],
)

py_library(
    name = "harbor_catalog",
    srcs = ["harbor_catalog.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
    "//experimental/overhead_matching/swag/data:landmark_schema",
        requirement("numpy"),
        requirement("pandas"),
        requirement("pyarrow"),
        requirement("shapely"),
    ],
)

py_test(
    name = "harbor_catalog_test",
    srcs = ["harbor_catalog_test.py"],
    deps = [
    "//experimental/overhead_matching/swag/data:landmark_schema",
        requirement("numpy"),
        ":harbor_catalog",
    ],
)

py_library(
    name = "bearing_matcher",
    srcs = ["bearing_matcher.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":harbor_catalog",
    ],
)

py_test(
    name = "bearing_matcher_test",
    srcs = ["bearing_matcher_test.py"],
    deps = [":bearing_matcher"],
)

py_binary(
    name = "run_index",
    srcs = ["run_index.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_binary(
    name = "m9_match_landmarks",
    srcs = ["m9_match_landmarks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pandas"),
        requirement("pyarrow"),
        "//experimental/overhead_matching/swag/data:landmark_schema",
        ":bearing_matcher",
        ":harbor_catalog",
        ":semantic_audit",
    ],
)

py_binary(
    name = "m10_match_viewer",
    srcs = ["m10_match_viewer.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_binary(
    name = "bow_calibration",
    srcs = ["bow_calibration.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":pano_geometry",
    ],
)

py_binary(
    name = "m6_merge_tracks",
    srcs = ["m6_merge_tracks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pillow"),
        ":semantic_audit",
        ":track_merge",
        "//experimental/overhead_matching/swag/landmark_filtering:ingest",
        "//experimental/overhead_matching/swag/landmark_filtering:pipeline_config",
    ],
)

py_binary(
    name = "keyframe_viewer",
    srcs = ["keyframe_viewer.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":pano_geometry",
        ":semantic_audit",
        ":track_builder",
        ":viz_common",
        "//experimental/overhead_matching/swag/landmark_filtering:ingest",
        "//experimental/overhead_matching/swag/landmark_filtering:pipeline_config",
    ],
)

py_binary(
    name = "m3_run_diff",
    srcs = ["m3_run_diff.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_binary(
    name = "m3_track_viewer",
    srcs = ["m3_track_viewer.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":pano_geometry",
        ":range_runner",
        ":track_builder",
        ":viz_common",
    ],
)

py_binary(
    name = "m2_sam_tracking",
    srcs = ["m2_sam_tracking.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        requirement("sam-2"),
        requirement("scipy"),
        requirement("torch"),
        ":heading",
        ":pano_geometry",
        ":video_frames",
        ":viz_common",
        "//common/torch:load_torch_deps",
        "//experimental/overhead_matching/swag/landmark_filtering:ingest",
        "//experimental/overhead_matching/swag/landmark_filtering:pipeline_config",
    ],
)

py_binary(
    name = "m0_render_boxes",
    srcs = ["m0_render_boxes.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":pano_geometry",
        ":viz_common",
        "//experimental/overhead_matching/swag/landmark_filtering:ingest",
        "//experimental/overhead_matching/swag/landmark_filtering:pipeline_config",
    ],
)

py_binary(
    name = "m11_localization_export",
    srcs = ["m11_localization_export.py"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        "//common/python:serialization",
        "//experimental/overhead_matching/swag/bearing_only_localization:geodesy",
        "//experimental/overhead_matching/swag/bearing_only_localization:structs",
        ":harbor_catalog",
    ],
)
