
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
    ":swag_model_input",
    ":swag_config_types",
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

py_library(
  name = "swag_model_input",
  srcs = ["swag_model_input.py"],
  deps = [
    requirement("torch"),
    "//common/torch:load_torch_deps",
  ]
)

py_library(
  name = "semantic_segment_extractor",
  srcs = ["semantic_segment_extractor.py"],
  deps = [
    requirement("torch"),
    requirement("msgspec"),
    requirement("sam-2"),
    requirement("huggingface_hub"),
    requirement("open-clip-torch"),
    requirement("ipdb"),
    requirement("pillow"),
    "//common/torch:load_torch_deps",
    ":swag_model_input",
    ":swag_config_types",
  ]
)

py_test(
  name = "semantic_segment_extractor_test",
  srcs = ["semantic_segment_extractor_test.py"],
  data = ["@vigor_snippet//:files"],
  deps = [
    requirement("PyGObject"),
    requirement("supervision"),
    requirement("matplotlib"),
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    ":semantic_segment_extractor",
  ]
)

py_library(
  name = "swag_config_types",
  srcs = ["swag_config_types.py"],
  deps = [
    requirement("msgspec"),
  ]
)
