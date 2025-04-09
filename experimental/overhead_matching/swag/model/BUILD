
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
