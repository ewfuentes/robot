load("//common/proto:proto.bzl", "multi_proto_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@pip//:requirements.bzl", "requirement")

cc_library(
  name = "proper_noun_matcher",
  srcs = ["proper_noun_matcher.cc"],
  hdrs = ["proper_noun_matcher.hh"],
  deps = ["@bs_thread_pool"],
  visibility = ["//visibility:public"],
)

pybind_extension(
  name = "proper_noun_matcher_python",
  srcs = ["proper_noun_matcher_python.cc"],
  deps = [":proper_noun_matcher"],
  visibility = ["//visibility:public"],
)

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
    "//common/geometry:gpu_geometry_collection",
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
    "//common/geometry:gpu_geometry_collection",
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

py_library(
  name = "combined_observation_likelihood",
  srcs = ["combined_observation_likelihood.py"],
  visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
  deps = [
    requirement("torch"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/filter:particle_filter",
  ]
)

py_test(
  name = "combined_observation_likelihood_test",
  srcs = ["combined_observation_likelihood_test.py"],
  deps = [
    requirement("torch"),
    "//common/torch:load_torch_deps",
    ":combined_observation_likelihood",
    ":swag_algorithm",
  ]
)

py_test(
  name = "proper_noun_matcher_test",
  srcs = ["proper_noun_matcher_test.py"],
  data = [":proper_noun_matcher_python.so"],
  deps = [
    requirement("numpy"),
  ],
)

py_library(
  name = "osm_tag_similarity",
  srcs = ["osm_tag_similarity.py"],
  data = [":proper_noun_matcher_python.so"],
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
    "//common/python:__subpackages__",
  ],
  deps = [
    requirement("numpy"),
    requirement("torch"),
    requirement("tqdm"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
  ],
)

py_library(
  name = "convergence_metrics",
  srcs = ["convergence_metrics.py"],
  visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
  deps = [
    requirement("torch"),
    "//common/gps:web_mercator",
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/filter:histogram_belief",
  ],
)

py_test(
  name = "convergence_metrics_test",
  srcs = ["convergence_metrics_test.py"],
  deps = [
    requirement("torch"),
    "//common/gps:web_mercator",
    "//common/math:haversine",
    "//common/torch:load_torch_deps",
    ":convergence_metrics",
    "//experimental/overhead_matching/swag/filter:histogram_belief",
  ],
)
