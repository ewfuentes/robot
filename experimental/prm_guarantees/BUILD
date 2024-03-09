
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

cc_binary(
  name="annoy_precision",
  srcs = ["annoy_precision.cc"],
  deps = ["@annoy"],
)

cc_test(
  name = "nearest_neighbor_graph_test",
  srcs = ["nearest_neighbor_graph_test.cc"],
  tags = ["manual"],
  linkopts = ["-ltbb"],
  deps = [
    "@annoy",
    "@com_google_googletest//:gtest_main",
  ]
)

py_library(
  name = "nearest_neighbors",
  srcs = ["nearest_neighbors.py"],
  deps = [
    requirement("triton"),
    requirement("torch")
  ]
)

py_test(
  name = "nearest_neighbors_test",
  srcs = ["nearest_neighbors_test.py"],
  deps = [
    ":nearest_neighbors",
    requirement('numpy')
  ]
)
