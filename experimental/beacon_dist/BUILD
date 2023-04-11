load("@pip//:requirements.bzl", "requirement")

py_binary(
  name="generate_data",
  srcs=["generate_data.py"],
  deps=[
    requirement("opencv-python"),
    requirement("Wand"),
    requirement("numpy"),
    requirement("tqdm"),
    requirement("IPython"),
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