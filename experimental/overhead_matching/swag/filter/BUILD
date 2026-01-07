load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "histogram_belief",
    srcs = ["histogram_belief.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("torch"),
        "//common/gps:web_mercator",
        "//common/torch:load_torch_deps",
        ":particle_filter",
    ],
)

py_test(
    name = "histogram_belief_test",
    srcs = ["histogram_belief_test.py"],
    deps = [
        requirement("torch"),
        "//common/torch:load_torch_deps",
        ":histogram_belief",
    ],
)

py_library(
  name = "particle_filter",
  srcs = ["particle_filter.py"],
  visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
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
