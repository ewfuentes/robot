
load("@pip//:requirements.bzl", "requirement")

py_binary(
  name = "svgd_sandbox",
  srcs = ["svgd_sandbox.py"],
  deps = [
    "//common/torch",
    requirement('matplotlib'),
    requirement('tqdm'),
    requirement('IPython'),
  ]
)
