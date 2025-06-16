
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "vigor_dataset",
  srcs = ["vigor_dataset.py"],
  visibility = ["//experimental/overhead_matching/swag:__subpackages__",
                "//common/python:__subpackages__"],
  deps = [
    requirement("numpy"),
    requirement("torch"),
    requirement("torchvision"),
    requirement("pandas"),
    requirement("scipy"),
    requirement("matplotlib"),
    requirement("PyGObject"),
    requirement("pytorch_memlab"),
    "//common/torch:load_torch_deps",
    "//common/math:haversine",
  ]
)

py_test(
  name = "vigor_dataset_test",
  srcs = ["vigor_dataset_test.py"],
  data=["@vigor_snippet//:files"],
  deps = [
    requirement("numpy"),
    requirement("Pillow"),
    requirement("torch"),
    "//common/torch:load_torch_deps",
    ":vigor_dataset",
  ]
)

py_library(
  name = "satellite_embedding_database",
  srcs = ["satellite_embedding_database.py"],
  visibility = ["//experimental/overhead_matching/swag:__subpackages__",
                "//common/python:__subpackages__"],
  deps = [
    requirement("torch"),
    requirement("torchvision"),
    requirement("tqdm"),
    "//common/torch:load_torch_deps",
  ]
)

py_test(
  name = "satellite_embedding_database_test",
  srcs = ["satellite_embedding_database_test.py"],
  data=["@vigor_snippet//:files"],
  deps = [
    ":satellite_embedding_database",
    ":vigor_dataset",
  ]
)
