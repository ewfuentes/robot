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
    requirement("torch")
  ]
)

py_test(
  name = "model_test",
  srcs = ["model_test.py"],
  deps = [
    ":model"
  ]
)

py_library(
  name = "utils",
  srcs = ["utils.py"],
  deps = [
    requirement("torch")
  ]
)

py_test(
  name = "utils_test",
  srcs = ["utils_test.py"],
  deps = [
    ":utils",
    requirement("numpy"),
  ]
)