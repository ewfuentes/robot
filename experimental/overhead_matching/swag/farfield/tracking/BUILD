load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "tracklets",
    srcs = ["tracklets.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "tracklets_test",
    srcs = ["tracklets_test.py"],
    deps = [
        requirement("numpy"),
        ":tracklets",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "build_bearing_observations_lib",
    srcs = ["build_bearing_observations.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":tracklets",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
        "//experimental/overhead_matching/swag/farfield:stage_reuse",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
    ],
)

py_binary(
    name = "build_bearing_observations",
    srcs = ["build_bearing_observations.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":build_bearing_observations_lib"],
)

py_test(
    name = "build_bearing_observations_test",
    srcs = ["build_bearing_observations_test.py"],
    deps = [
        ":build_bearing_observations_lib",
        ":tracklets",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:testing",
    ],
)

py_library(
    name = "perf_profile",
    srcs = ["perf_profile.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = ["//common/torch:load_torch_deps"],
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
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "track_builder",
    srcs = ["track_builder.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
        ":perf_profile",
        "//experimental/overhead_matching/swag/farfield:geometry",
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
        ":perf_profile",
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
        ":perf_profile",
        ":sam_backend",
        ":track_builder",
        ":video_frames",
        ":viz_common",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield/calibration:heading",
    ],
)

py_library(
    name = "run_tracking_lib",
    srcs = ["run_tracking.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python"),
        requirement("pillow"),
        ":keyframe_viewer_lib",
        ":perf_profile",
        ":range_runner",
        ":track_builder",
        ":viz_common",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
        "//experimental/overhead_matching/swag/farfield/viewers:page",
    ],
)

py_binary(
    name = "run_tracking",
    srcs = ["run_tracking.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":run_tracking_lib"],
)

py_test(
    name = "run_tracking_test",
    srcs = ["run_tracking_test.py"],
    deps = [
        requirement("numpy"),
        ":range_runner",
        ":run_tracking_lib",
        ":track_builder",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:testing",
    ],
)

py_library(
    name = "keyframe_viewer_lib",
    srcs = ["keyframe_viewer.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":perf_profile",
        ":track_builder",
        ":viz_common",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield/viewers:indexes",
        "//experimental/overhead_matching/swag/farfield/viewers:page",
    ],
)

py_binary(
    name = "keyframe_viewer",
    srcs = ["keyframe_viewer.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":keyframe_viewer_lib"],
)

py_test(
    name = "keyframe_viewer_test",
    srcs = ["keyframe_viewer_test.py"],
    deps = [
        ":keyframe_viewer_lib",
        ":track_builder",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:testing",
    ],
)

py_library(
    name = "semantic_audit",
    srcs = ["semantic_audit.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pydantic"),
        ":track_builder",
        ":viz_common",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield/extraction:prompts",
    ],
)

py_test(
    name = "semantic_audit_test",
    srcs = ["semantic_audit_test.py"],
    deps = [
        requirement("google-genai"),
        ":semantic_audit",
        ":track_builder",
        "//experimental/overhead_matching/swag/farfield:dataset",
    ],
)

py_library(
    name = "audit_requests_lib",
    srcs = ["audit_requests.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":keyframe_viewer_lib",
        ":semantic_audit",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
        "//experimental/overhead_matching/swag/farfield:stage_reuse",
        "//experimental/overhead_matching/swag/farfield/extraction:vertex_batch_manager_lib",
        "//experimental/overhead_matching/swag/farfield/viewers:page",
    ],
)

py_binary(
    name = "audit_requests",
    srcs = ["audit_requests.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":audit_requests_lib"],
)

py_test(
    name = "audit_requests_test",
    srcs = ["audit_requests_test.py"],
    deps = [
        ":audit_requests_lib",
        ":semantic_audit",
        ":track_builder",
        ":tracklets",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:pipeline_lib",
        "//experimental/overhead_matching/swag/farfield:testing",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
        "//experimental/overhead_matching/swag/farfield/extraction:prompts",
    ],
)

py_library(
    name = "audit_review_lib",
    srcs = ["audit_review.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":keyframe_viewer_lib",
        ":semantic_audit",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
        "//experimental/overhead_matching/swag/farfield/viewers:page",
    ],
)

py_binary(
    name = "audit_review",
    srcs = ["audit_review.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":audit_review_lib"],
)

py_test(
    name = "audit_review_test",
    srcs = ["audit_review_test.py"],
    deps = [
        ":audit_requests_lib",
        ":audit_review_lib",
        ":track_builder",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:testing",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
    ],
)
