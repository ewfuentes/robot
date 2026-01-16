
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "vigor_dataset",
  srcs = ["vigor_dataset.py"],
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
    "//common/python:__subpackages__",
  ],
  deps = [
    requirement("numpy"),
    requirement("torch"),
    requirement("torchvision"),
    requirement("pandas"),
    requirement("geopandas"),
    requirement("shapely"),
    requirement("pyarrow"),
    requirement("scipy"),
    requirement("matplotlib"),
    requirement("psutil"),
    requirement("PyGObject"),
    requirement("lmdb"),
    requirement("msgspec"),
    "//common/torch:load_torch_deps",
    "//common/math:haversine",
    "//common/gps:web_mercator",
    "//experimental/overhead_matching/swag/scripts:distances",
    "//experimental/overhead_matching/swag/model:swag_model_input_output",
    "//experimental/overhead_matching/swag/model:swag_config_types",
    "//experimental/overhead_matching/swag/model:semantic_landmark_utils",
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
    requirement("pandas"),
    requirement("geopandas"),
    "//common/torch:load_torch_deps",
    ":vigor_dataset",
    "//experimental/overhead_matching/swag/scripts:distances",
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

py_library(
  name = "alphaearth_registry",
  srcs = ["alphaearth_registry.py"],
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
    "//common/python:__subpackages__",
  ],
  deps = [
    requirement("rasterio"),
    requirement("pyproj"),
  ],
)

py_test(
  name = "alphaearth_registry_test",
  srcs = ["alphaearth_registry_test.py"],
  data = ["@alphaearth_snippet//:files"],
  deps = [":alphaearth_registry"],
)

py_library(
  name = "osm_sentence_generator",
  srcs = ["osm_sentence_generator.py"],
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
    "//common/python:__subpackages__",
  ],
  deps = [],
)

py_test(
  name = "osm_sentence_generator_test",
  srcs = ["osm_sentence_generator_test.py"],
  deps = [":osm_sentence_generator"],
)

py_library(
  name = "sentence_dataset",
  srcs = ["sentence_dataset.py"],
  visibility = [
    "//experimental/overhead_matching/swag:__subpackages__",
  ],
  deps = [
    requirement("torch"),
    ":osm_sentence_generator",
  ],
)
