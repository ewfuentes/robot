load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

py_library(
  name="clevr_dataset",
  srcs=["clevr_dataset.py"],
  visibility=["//experimental/overhead_matching/learned:__subpackages__"],
  deps=[
    "//common/torch:load_torch_deps",
  ]
)

py_test(
  name="clevr_dataset_test",
  srcs=["clevr_dataset_test.py"],
  data=["@clevr_test_set//:files"],
  deps=[
    ":clevr_dataset",
  ]
)

