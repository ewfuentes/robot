load("@pip//:requirements.bzl", "requirement")

py_binary(
  name="generate_letter_dataset",
  srcs=["generate_letter_dataset.py"],
  visibility = ["//common/python:__subpackages__"],
  deps=[
    requirement("opencv-python"),
    requirement("Wand"),
    requirement("numpy"),
    requirement("tqdm"),
    ":utils",
  ]
)

py_binary(
  name = "generate_ycb_dataset",
  srcs = ["generate_ycb_dataset.py"],
  visibility = ["//common/python:__subpackages__"],
  deps = [
    requirement("drake"),
    requirement("numpy"),
    requirement("opencv-python"),
    requirement("tqdm"),
  ]
)

py_library(
  name = "model",
  srcs = ["model.py"],
  visibility = ["//common/python:__subpackages__"],
  deps = [
    requirement("torch"),
  ]
)

py_test(
  name = "model_test",
  srcs = ["model_test.py"],
  deps = [
    ":test_helpers",
    ":model"
  ]
)

py_library(
  name = "utils",
  srcs = ["utils.py"],
  visibility = ["//common/python:__subpackages__"],
  deps = [
    requirement("torch"),
    requirement("numpy"),
  ]
)

py_library(
  name = "test_helpers",
  srcs = ["test_helpers.py"],
  deps = [
    requirement("numpy"),
    ":utils",
  ]
)

py_test(
  name = "utils_test",
  srcs = ["utils_test.py"],
  deps = [
    requirement("numpy"),
    ":test_helpers",
    ":utils",
  ]
)

py_binary(
  name = "train",
  srcs = ["train.py"],
  visibility = ["//common/python:__subpackages__"],
  deps = [
    requirement("torch"),
    requirement("IPython"),
    ":utils",
    ":model",
  ]
)

