load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_binary(
  name = "jupyter_notebook",
  srcs = ["jupyter_notebook.py"],
  deps = [
    requirement("notebook"),
    requirement("matplotlib"),
    requirement("Pillow"),
    requirement("numpy"),
  ]
)