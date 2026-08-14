load("@pip//:requirements.bzl", "extra_requirement", "requirement")

package(default_visibility = ["//experimental/map_estimation:__subpackages__"])

py_library(
    name = "av2_source",
    srcs = ["av2_source.py"],
    deps = [
        "//experimental/map_estimation/data:argoverse_layout",
        requirement("av2"),
    ],
)

py_test(
    name = "av2_source_test",
    size = "small",
    srcs = ["av2_source_test.py"],
    deps = [":av2_source"],
)

py_library(
    name = "av2_scene",
    srcs = ["av2_scene.py"],
    deps = [
        ":av2_source",
        # NOT requirement("rerun-sdk"): the wheel hides its package one directory down and
        # relies on a .pth file Bazel does not read. See third_party/python/extra_rerun_targets.bzl.
        extra_requirement("rerun-sdk", "rerun"),
    ],
)

py_test(
    name = "av2_scene_test",
    size = "small",
    srcs = ["av2_scene_test.py"],
    deps = [":av2_scene"],
)

py_binary(
    name = "view_log",
    srcs = ["view_log.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":av2_scene",
        ":av2_source",
        "//experimental/map_estimation/data:argoverse_layout",
    ],
)
