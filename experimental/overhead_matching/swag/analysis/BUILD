
load("@pip//:requirements.bzl", "requirement")

py_binary(
  name="compare_histogram_evaluations",
  srcs=["compare_histogram_evaluations.py"],
  deps=[
    requirement("torch"),
    requirement("matplotlib"),
    requirement("numpy"),
    "//common/torch:load_torch_deps",
  ]
)

py_binary(
  name="analyze_fusion_diagnostics",
  srcs=["analyze_fusion_diagnostics.py"],
  deps=[
    requirement("torch"),
    requirement("numpy"),
    requirement("pandas"),
    requirement("matplotlib"),
    requirement("marimo"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
  ]
)

py_binary(
  name="visualize_fusion_diagnostics",
  srcs=["visualize_fusion_diagnostics.py"],
  deps=[
    requirement("torch"),
    requirement("numpy"),
    requirement("dash"),
    requirement("plotly"),
    requirement("pillow"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
  ]
)
