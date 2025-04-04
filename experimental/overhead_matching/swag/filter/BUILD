load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "particle_filter",
  srcs = ["particle_filter.py"],
  deps = [
    requirement("torch"),
    "//common/torch:load_torch_deps",
  ]
)

py_test(
  name = "particle_filter_test",
  srcs = ["particle_filter_test.py"],
  deps = [
    requirement("torch"),
    requirement("numpy"),
    requirement("matplotlib"),
    "//common/torch:load_torch_deps",
    ":particle_filter",
  ],
)
