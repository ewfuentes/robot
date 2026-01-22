
load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:packaging.bzl", "py_wheel", "py_package")

py_library(
  name="pairing",
  srcs=["pairing.py"],
  visibility=["//common/python:__subpackages__"],
)

py_test(
  name="pairing_test",
  srcs=["pairing_test.py"],
  deps=[
    ":pairing",
    "//common/torch:load_torch_deps",
  ]
)

py_test(
  name="distances_test",
  srcs=["distances_test.py"],
  deps=[
    ":distances",
    "//common/torch:load_torch_deps",
    requirement("torch"),
    requirement("msgspec"),
    "//common/python:serialization",
  ]
)

py_library(
  name="losses",
  srcs=["losses.py"],
  visibility=["//common/python:__subpackages__"],
  deps=[
    ":pairing",
    ":distances",
    requirement("torch"),
    requirement("msgspec"),
    "//common/python:serialization",
    "//common/torch:load_torch_deps",
  ]
)

py_library(
  name="distances",
  srcs=["distances.py"],
  visibility=["//common/python:__subpackages__", "//experimental/overhead_matching/swag:__subpackages__"],
  deps=[
    "//common/python:serialization",
    requirement("msgspec"),
    requirement("torch"),
  ]
)
py_library(
  name="logging_utils",
  srcs=["logging_utils.py"],
  visibility=["//common/python:__subpackages__"],
  deps=[
    requirement("torch"),
  ]
)

py_library(
  name = "lr_sweep",
  srcs = ["lr_sweep.py"],
  visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
  deps = [
    requirement("torch"),
    requirement("numpy"),
    requirement("matplotlib"),
  ]
)

py_library(
  name = "model_inspector",
  srcs = ["model_inspector.py"],
  visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
  deps = [
    requirement("torch"),
    ":pairing",
    "//experimental/overhead_matching/swag/model:swag_model_input_output",
  ]
)

py_binary(
  name="train",
  srcs=["train.py"],
  visibility=["//common/python:__subpackages__", "//common/tools/lambda_cloud:__subpackages__"],
  deps=[
    ":logging_utils",
    ":lr_sweep",
    ":model_inspector",
    ":pairing",
    ":losses",
    ":distances",
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
  visibility=["//common/python:__subpackages__"],
  deps=[
    requirement("torch"),
    requirement("tqdm"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/evaluation:swag_algorithm",
    "//experimental/overhead_matching/swag/evaluation:wag_config_proto_py",
    "//experimental/overhead_matching/swag/evaluation:evaluate_swag",
    "//experimental/overhead_matching/swag/evaluation:observation_likelihood",
    "//experimental/overhead_matching/swag/model:patch_embedding",
    "//experimental/overhead_matching/swag/model:swag_patch_embedding",
  ]
)

py_binary(
  name="grid_search_combined_likelihood",
  srcs=["grid_search_combined_likelihood.py"],
  visibility=["//common/python:__subpackages__"],
  deps=[
    requirement("torch"),
    requirement("tqdm"),
    requirement("msgspec"),
    "//common/torch:load_torch_deps",
    "//common/torch:load_and_save_models",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/evaluation:swag_algorithm",
    "//experimental/overhead_matching/swag/evaluation:wag_config_proto_py",
    "//experimental/overhead_matching/swag/evaluation:evaluate_swag",
    "//experimental/overhead_matching/swag/evaluation:combined_observation_likelihood",
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
    ":losses",
    ":distances",
    "//common/python:serialization",
  ]
)

py_binary(
  name = "extract_landmarks_for_vigor_dataset",
  srcs = ["extract_landmarks_for_vigor_dataset.py"],
  deps = [
    requirement("osmnx"),
    requirement("pandas"),
    requirement("pyarrow"),
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//common/gps:web_mercator",
  ]
)

py_binary(
  name = "extract_landmarks_historical",
  srcs = ["extract_landmarks_historical.py"],
  data = [
    "//common/openstreetmap:extract_landmarks_python.so",
  ],
  deps = [
    requirement("geopandas"),
    requirement("shapely"),
    requirement("pyarrow"),
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
    "//common/python:serialization",
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    ":train",
  ]
)

py_binary(
  name="panorama_to_pinhole",
  srcs=["panorama_to_pinhole.py"],
  deps=[
    requirement("numpy"),
    requirement("scipy"),
    requirement("pillow"),
    requirement("tqdm"),
  ]
)

py_binary(
  name="panorama_viewer",
  srcs=["panorama_viewer.py"],
  deps=[
    requirement("flask"),
    requirement("geopandas"),
    requirement("shapely"),
    requirement("pandas"),
    requirement("ipdb"),
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/model:semantic_landmark_utils",
    "//common/gps:web_mercator",
  ]
)

py_binary(
  name="reexport_geojson",
  srcs=["reexport_geojson.py"],
  deps=[
    requirement("geopandas"),
    requirement("pyarrow"),
  ]
)

py_binary(
  name="visualize_observation_likelihood",
  srcs=["visualize_observation_likelihood.py"],
  deps=[
    requirement("torch"),
    requirement("numpy"),
    requirement("dash"),
    requirement("plotly"),
    requirement("geopandas"),
    requirement("shapely"),
    requirement("msgspec"),
    "//common/torch:load_torch_deps",
    "//common/torch:load_and_save_models",
    "//common/gps:web_mercator",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/evaluation:evaluate_swag",
    "//experimental/overhead_matching/swag/evaluation:observation_likelihood",
    "//experimental/overhead_matching/swag/filter:particle_filter",
    "//experimental/overhead_matching/swag/model:patch_embedding",
    "//experimental/overhead_matching/swag/model:swag_patch_embedding",
  ]
)

py_binary(
  name="generate_city_bboxes",
  srcs=["generate_city_bboxes.py"],
  deps=[
    requirement("pyyaml"),
  ]
)

py_binary(
  name="extract_landmarks_to_sqlite",
  srcs=["extract_landmarks_to_sqlite.py"],
  data = [
    "//common/openstreetmap:extract_landmarks_python.so",
  ],
  deps=[
    requirement("pyyaml"),
    requirement("shapely"),
  ]
)

py_library(
  name = "sentence_configs",
  srcs = ["sentence_configs.py"],
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
  ],
  deps = [
    requirement("msgspec"),
    requirement("pyyaml"),
    "//common/python:serialization",
  ],
)

py_test(
  name = "sentence_configs_test",
  srcs = ["sentence_configs_test.py"],
  deps = [
    ":sentence_configs",
    requirement("msgspec"),
    requirement("pyyaml"),
  ],
)

py_library(
  name = "sentence_losses",
  srcs = ["sentence_losses.py"],
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
  ],
  deps = [
    requirement("torch"),
    "//experimental/overhead_matching/swag/data:sentence_dataset",
  ],
)

py_binary(
  name = "build_tag_vocabularies",
  srcs = ["build_tag_vocabularies.py"],
  deps = [],
)

py_binary(
  name = "train_sentence_embeddings",
  srcs = ["train_sentence_embeddings.py"],
  deps = [
    requirement("torch"),
    requirement("tensorboard"),
    requirement("tqdm"),
    requirement("numpy"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:sentence_dataset",
    "//experimental/overhead_matching/swag/data:paired_sentence_dataset",
    "//experimental/overhead_matching/swag/model:sentence_embedding_model",
    ":sentence_configs",
    ":sentence_losses",
  ],
)

py_binary(
  name="vertex_batch_manager",
  srcs=["vertex_batch_manager.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
  deps=[
    requirement("google-genai"),
    requirement("google-cloud-storage"),
  ]
)

py_binary(
  name="create_embeddings_with_gemini",
  srcs=["create_embeddings_with_gemini.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
  deps=[
    requirement("tqdm"),
    requirement("torch"),
    requirement("google-genai"),
    "//common/torch:load_torch_deps",
  ]
)
