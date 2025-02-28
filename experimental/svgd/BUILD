
load("@pip//:requirements.bzl", "requirement")

py_binary(
  name = "svgd_sandbox",
  srcs = ["svgd_sandbox.py"],
  deps = [
    "//common/torch:load_torch_deps",
    requirement('matplotlib'),
    requirement('tqdm'),
    requirement('IPython'),
  ]
)
