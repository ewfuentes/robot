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