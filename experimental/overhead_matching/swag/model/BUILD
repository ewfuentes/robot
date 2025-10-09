
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
    ":swag_config_types",
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
    ":alphaearth_extractor",
    ":semantic_landmark_extractor",
    ":synthetic_landmark_extractor",
    ":semantic_segment_extractor",
    ":absolute_position_extractor",
    ":swag_config_types",
    ":swag_model_input_output",
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
  name = "swag_model_input_output",
  srcs = ["swag_model_input_output.py"],
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
    "//common/python:__subpackages__",
  ],
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
    ":swag_model_input_output",
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
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
    "//common/python:__subpackages__",
  ],
  deps = [
    requirement("msgspec"),
    "//common/python:serialization",
  ]
)

py_library(
  name = "alphaearth_extractor",
  srcs = ["alphaearth_extractor.py"],
  deps = [
    requirement("torch"),
    ":swag_model_input_output",
    ":swag_config_types",
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:alphaearth_registry",
  ]
)

py_test(
  name = "alphaearth_extractor_test",
  data = ["@alphaearth_snippet//:files"],
  srcs = ["alphaearth_extractor_test.py"],
  deps = [
    "//common/gps:web_mercator",
    ":alphaearth_extractor",
  ]
)

py_binary(
  name = "semantic_landmark_extractor",
  srcs = ["semantic_landmark_extractor.py"],
  deps = [
    requirement("torch"),
    requirement("numpy"),
    requirement("sentence-transformers"),
    requirement("shapely"),
    requirement("openai"),
    requirement("geopandas"),
    requirement("ipdb"),
    requirement("tqdm"),
    "//common/torch:load_torch_deps",
    "//common/ollama:pyollama",
    ":swag_model_input_output",
    ":swag_config_types",
  ]
)

py_test(
  name = "semantic_landmark_extractor_test",
  srcs = ["semantic_landmark_extractor_test.py"],
  data = ["@vigor_snippet//:files"],
  deps = [
    requirement("shapely"),
    ":semantic_landmark_extractor",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
  ]
)

py_library(
  name = "synthetic_landmark_extractor",
  srcs = ["synthetic_landmark_extractor.py"],
  deps = [
    requirement("torch"),
    requirement("ipdb"),
    "//common/torch:load_torch_deps",
    ":swag_model_input_output",
    ":swag_config_types",
  ]
)

py_test(
  name = "synthetic_landmark_extractor_test",
  srcs = ["synthetic_landmark_extractor_test.py"],
  data = ["@vigor_snippet//:files"],
  deps = [
    ":synthetic_landmark_extractor",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
  ]
)


py_library(
  name = "absolute_position_extractor",
  srcs = ["absolute_position_extractor.py"],
  deps = [
    requirement("torch"),
    requirement("numpy"),
    "//common/torch:load_torch_deps",
    ":swag_model_input_output",
    ":swag_config_types",
  ]
)

py_test(
  name = "absolute_position_extractor_test",
  srcs = ["absolute_position_extractor_test.py"],
  data = ["@vigor_snippet//:files"],
  deps = [
    ":absolute_position_extractor",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
  ]
)
