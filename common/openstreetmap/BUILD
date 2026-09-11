load("@pip//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_test")

cc_test(
    name = "openstreetmap_test",
    srcs = ["openstreetmap_test.cc"],
    copts = ["-std=c++17"],
    data = ["@openstreetmap_snippet//:files"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@fmt",
        "@libosmium",
    ],
)

cc_library(
    name = "extract_landmarks",
    srcs = ["extract_landmarks.cc"],
    hdrs = ["extract_landmarks.hh"],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"],
    deps = [
        "@fmt",
        "@libosmium",
    ],
)

cc_test(
    name = "extract_landmarks_test",
    srcs = ["extract_landmarks_test.cc"],
    copts = ["-std=c++17"],
    data = ["@openstreetmap_snippet//:files"],
    deps = [
        ":extract_landmarks",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "extract_landmarks_python",
    srcs = ["extract_landmarks_python.cc"],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"],
    deps = [":extract_landmarks"],
)

py_test(
    name = "extract_landmarks_python_test",
    srcs = ["extract_landmarks_python_test.py"],
    data = [
        ":extract_landmarks_python.so",
        "@openstreetmap_snippet//:files",
    ],
    deps = [
        requirement("geopandas"),
        requirement("shapely"),
        requirement("pyarrow"),
    ],
)
