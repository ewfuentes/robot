load("//common/proto:proto.bzl", "multi_proto_library")
load("@pip//:requirements.bzl", "requirement")

multi_proto_library(
  name = "wag_config_proto",
  srcs = ["wag_config.proto"],
  visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_library(
  name = "swag_algorithm",
  srcs = ["swag_algorithm.py"],
  visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
  deps = [
    requirement("torch"),
    requirement("pandas"),
    requirement("torchvision"),
    ":wag_config_proto_py",
    "//common/torch:load_torch_deps",
    "//common/torch:load_and_save_models",
    "//experimental/overhead_matching/swag/filter:particle_filter",
  ]
)

py_library(
  name = "evaluate_swag",
  srcs = ["evaluate_swag.py"],
  visibility = ["//common/python:__subpackages__", "//experimental/overhead_matching/swag:__subpackages__"],
  deps = [
    requirement("torch"),
    requirement("pandas"),
    requirement("tqdm"),
    requirement("torchvision"),
    requirement("torch_kdtree"),
    ":wag_config_proto_py",
    "//common/torch:load_torch_deps",
    "//common/torch:load_and_save_models",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/evaluation:swag_algorithm",
    "//experimental/overhead_matching/swag/data:satellite_embedding_database",
  ]
)

py_test(
  name = "evaluate_swag_test",
  srcs = ["evaluate_swag_test.py"],
  data=["@vigor_snippet//:files"],
  deps = [
    requirement("torch"),
    "//common/torch:load_torch_deps",
    ":evaluate_swag",
  ],
)

py_library(
  name = "observation_likelihood",
  srcs = ["observation_likelihood.py"],
  visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
  deps = [
    requirement("torch"),
    requirement("geopandas"),
    "//common/torch:load_torch_deps",
    "//common/gps:web_mercator",
    "//experimental/overhead_matching/swag/model:semantic_landmark_utils",
  ]
)

py_test(
  name = "observation_likelihood_test",
  srcs = ["observation_likelihood_test.py"],
  data = ["@vigor_snippet//:files"],
  deps = [
    requirement("torch"),
    requirement("pandas"),
    requirement("geopandas"),
    requirement("shapely"),
    "//common/torch:load_torch_deps",
    "//common/gps:web_mercator",
    ":observation_likelihood",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/model:semantic_landmark_utils",
  ]
)

py_test(
  name = "swag_algorithm_test",
  srcs = ["swag_algorithm_test.py"],
  deps = [
    requirement("torch"),
    "//common/torch:load_torch_deps",
    ":swag_algorithm",
    ":wag_config_proto_py",
  ]
)
