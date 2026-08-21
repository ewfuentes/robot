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
        ":perf_profile",
        ":range_runner",
        ":track_builder",
        ":viz_common",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
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
        ":run_tracking_lib",
        ":track_builder",
        "//experimental/overhead_matching/swag/farfield:dataset",
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
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
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
    ],
)
