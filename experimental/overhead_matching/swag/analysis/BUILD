
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
  name="correspondence_explorer",
  srcs=["correspondence_explorer.py"],
  deps=[
    requirement("flask"),
    requirement("numpy"),
    requirement("scipy"),
    requirement("torch"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/evaluation:correspondence_matching",
  ]
)

py_binary(
  name="correspondence_fusion_explorer",
  srcs=["correspondence_fusion_explorer.py"],
  deps=[
    requirement("marimo"),
    requirement("torch"),
    requirement("numpy"),
    "//common/torch:load_torch_deps",
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
