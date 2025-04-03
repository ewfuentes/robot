
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "vigor_dataset",
  srcs = ["vigor_dataset.py"],
  deps = [
    requirement("torch"),
    requirement("torchvision"),
    requirement("pandas"),
    requirement("scipy"),
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
    requirement("torch"),
    "//common/torch:load_torch_deps",
    ":vigor_dataset",
  ]
)
