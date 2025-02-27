
load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "torch",
  srcs = ["torch.py"],
  deps = [
    requirement("torch"),
    requirement("numpy"),
  ],
  visibility = ["//visibility:public"],
)

