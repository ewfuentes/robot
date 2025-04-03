
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "vigor_dataset",
  srcs = ["vigor_dataset.py"],
  deps = [
    requirement("torch"),
    requirement("pandas"),
    requirement("matplotlib"),
    requirement("PyGObject"),
    "//common/torch:load_torch_deps",
  ]
)

py_test(
  name = "vigor_dataset_test",
  srcs = ["vigor_dataset_test.py"],
  deps = [
    requirement("numpy"),
    requirement("Pillow"),
    ":vigor_dataset",
  ]
)
