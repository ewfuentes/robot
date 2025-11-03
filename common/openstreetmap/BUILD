load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")

cc_test(
  name = "openstreetmap_test",
  srcs = ["openstreetmap_test.cc"],
  data = ["@openstreetmap_snippet//:files"],
  deps = [
    "@fmt",
    "@libosmium",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "extract_landmarks",
  hdrs = ["extract_landmarks.h"],
  srcs = ["extract_landmarks.cc"],
  deps = [
    "@libosmium",
    "@fmt",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "extract_landmarks_test",
  srcs = ["extract_landmarks_test.cc"],
  data = ["@openstreetmap_snippet//:files"],
  deps = [
    ":extract_landmarks",
    "@com_google_googletest//:gtest_main",
  ]
)

pybind_extension(
  name = "extract_landmarks_python",
  srcs = ["extract_landmarks_python.cc"],
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
