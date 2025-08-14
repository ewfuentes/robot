
load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:packaging.bzl", "py_wheel", "py_package")

py_binary(
  name="train",
  srcs=["train.py"],
  visibility=["//common/python:__subpackages__"],
  deps=[
    requirement("torch"),
    requirement("tensorboard"),
    requirement("tqdm"),
    requirement("IPython"),
    requirement("msgspec"),
    requirement("pyyaml"),
    "//common/python:serialization",
    "//common/torch:load_torch_deps",
    "//common/torch:load_and_save_models",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/data:satellite_embedding_database",
    "//experimental/overhead_matching/swag/model:patch_embedding",
    "//experimental/overhead_matching/swag/model:swag_patch_embedding",
  ]
)

py_package(
  name="train_package",
  deps = [
    ":train",
  ]
)

py_wheel(
  name="train_wheel",
  distribution="overhead_matching_train",
  version = "0.0.1",
  platform = "manylinux2014_x86_64",
  tags = ["manual"],
  python_tag = select({
    "@rules_python//python/config_settings:is_python_3.12": "cp312"
  }),
  abi = select({
    "@rules_python//python/config_settings:is_python_3.12": "cp312"
  }),
  deps=[":train_package"],
)

py_binary(
  name="create_evaluation_paths",
  srcs=["create_evaluation_paths.py"],
  deps=[
    requirement("torch"),
    requirement("tqdm"),
    requirement("matplotlib"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
  ]
)
py_binary(
  name="evaluate_model_on_paths",
  srcs=["evaluate_model_on_paths.py"],
  deps=[
    requirement("torch"),
    requirement("tqdm"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/evaluation:swag_algorithm",
    "//experimental/overhead_matching/swag/evaluation:wag_config_proto_py",
    "//experimental/overhead_matching/swag/evaluation:evaluate_swag",
    "//experimental/overhead_matching/swag/model:patch_embedding",
    "//experimental/overhead_matching/swag/model:swag_patch_embedding",
  ]
)

py_binary(
  name="plot_path_evaluations",
  srcs=["plot_path_evaluations.py"],
  deps=[
    ":evaluate_model_on_paths",
    requirement("torch"),
    requirement("tqdm"),
    "//common/torch:load_torch_deps",
  ]
)

py_binary(
  name = "generate_training_configs",
  srcs = ["generate_training_configs.py"],
  deps = [
    requirement("msgspec"),
    ":train",
  ]
)

py_binary(
  name = "extract_landmarks_for_vigor_dataset",
  srcs = ["extract_landmarks_for_vigor_dataset.py"],
  deps = [
    requirement("osmnx"),
    requirement("pandas"),
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//common/gps:web_mercator",
  ]
)

py_binary(
  name="step_through_filter",
  srcs=["step_through_filter.py"],
  deps=[
    ":evaluate_model_on_paths",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/evaluation:swag_algorithm",
    "//experimental/overhead_matching/swag/evaluation:wag_config_proto_py",
    "//experimental/overhead_matching/swag/evaluation:evaluate_swag",
    "//experimental/overhead_matching/swag/model:patch_embedding",
    requirement("torch"),
    requirement("dash"),
    requirement("tqdm"),
    requirement("opencv-python"),
    "//common/torch:load_torch_deps",
  ]
)

py_binary(
  name="populate_tensor_cache",
  srcs=["populate_tensor_cache.py"],
  deps=[
    requirement("lmdb"),
    requirement("msgspec"),
    requirement("torch"),
    requirement("tqdm"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    ":train",
  ]
)
