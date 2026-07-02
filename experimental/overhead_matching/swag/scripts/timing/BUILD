
load("@pip//:requirements.bzl", "requirement")

py_binary(
  name="measure_pipeline_latency",
  srcs=["measure_pipeline_latency.py"],
  deps=[
    requirement("torch"),
    requirement("numpy"),
    requirement("ollama"),
    requirement("sentence-transformers"),
    requirement("google-genai"),
    "//common/torch:load_torch_deps",
    "//common/gps:web_mercator",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/data:landmark_correspondence_dataset",
    "//experimental/overhead_matching/swag/evaluation:correspondence_matching",
    "//experimental/overhead_matching/swag/evaluation:evaluate_swag",
    "//experimental/overhead_matching/swag/filter:histogram_belief",
    "//experimental/overhead_matching/swag/model:landmark_correspondence_model",
    "//experimental/overhead_matching/swag/model:semantic_landmark_extractor",
    "//experimental/overhead_matching/swag/scripts:evaluate_histogram_on_paths",
    "//experimental/overhead_matching/swag/scripts:ollama_osm_extraction_lib",
  ]
)

py_library(
  name="correspondence_scaling_common",
  srcs=["correspondence_scaling_common.py"],
  visibility=["//experimental/overhead_matching/swag:__subpackages__"],
  deps=[
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/model:additional_panorama_extractors",
    "//experimental/overhead_matching/swag/scripts:landmark_pairing_cli_lib",
  ]
)

py_binary(
  name="benchmark_correspondence_scaling_ollama",
  srcs=["benchmark_correspondence_scaling_ollama.py"],
  deps=[
    requirement("ollama"),
    ":correspondence_scaling_common",
    "//common/ollama:pyollama",
  ]
)
