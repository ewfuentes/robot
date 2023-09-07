
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "color_histogram",
    srcs = ["color_histogram.py"],
    deps = [requirement("torch")]
)

py_test(
    name = "color_histogram_test",
    srcs = ["color_histogram_test.py"],
    deps = [":color_histogram"]
)
