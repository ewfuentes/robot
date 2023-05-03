load("@pip//:requirements.bzl", "requirement")

py_binary(
  name="generate_data",
  srcs=["generate_data.py"],
  deps=[
    requirement("opencv-python"),
    requirement("Wand"),
    requirement("numpy"),
    requirement("tqdm"),
    ":utils",
  ]
)

py_library(
  name = "model",
  srcs = ["model.py"],
  deps = [
    requirement("numpy"),
    requirement("torch")
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
  deps = [
    requirement("torch"),
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
    ":test_helpers",
    ":utils",
  ]
)

py_binary(
  name = "train",
  srcs = ["train.py"],
  deps = [
    requirement("torch"),
    ":utils",
    ":model",
  ]
)
