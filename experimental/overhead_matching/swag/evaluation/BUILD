
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "evaluate_swag",
  srcs = ["evaluate_swag.py"],
  visibility = ["//common/python:__subpackages__"],
  deps = [
    requirement("torch"),
    requirement("pandas"),
    requirement("torchvision"),
    "//common/torch:load_torch_deps",
    "//common/torch:load_and_save_models",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
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
