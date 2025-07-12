
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "patch_embedding",
  srcs = ["patch_embedding.py"],
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
    "//common/python:__subpackages__",
  ],
  deps = [
    requirement("torch"),
    requirement("torchvision"),
    requirement("msgspec"),
    "//common/torch:load_torch_deps",
  ]
)

py_test(
  name = "patch_embedding_test",
  srcs = ["patch_embedding_test.py"],
  deps = [
    requirement("torch"),
    "//common/torch:load_torch_deps",
    ":patch_embedding",
  ],
)

py_library(
  name = "swag_patch_embedding",
  srcs = ["swag_patch_embedding.py"],
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
    "//common/python:__subpackages__",
  ],
  deps=[
    requirement("torch"),
    requirement("torchvision"),
    requirement("msgspec"),
    "//common/torch:load_torch_deps",
  ]
)

py_test(
  name = "swag_patch_embedding_test",
  srcs = ["swag_patch_embedding_test.py"],
  deps = [
    requirement("torch"),
    "//common/torch:load_torch_deps",
    ":swag_patch_embedding",
  ]
)
