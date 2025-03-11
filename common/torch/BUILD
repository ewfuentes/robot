
load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "load_torch_deps",
  srcs = ["load_torch_deps.py"],
  deps = [
    requirement("torch"),
  ],
  visibility = ["//visibility:public"],
)

py_library(
  name="load_and_save_models",
  srcs=["load_and_save_models.py"],
  visibility=["//visibility:public"],
  deps=[
    ":load_torch_deps",
    "//toolchain:git_info",
    requirement("numpy"),
  ]
)

py_binary(
    name="run_tensorboard",
    srcs=["run_tensorboard.py"],
    deps=[
        requirement("tensorboard"),
    ]
)

py_test(
  name="load_and_save_models_test",
  srcs=["load_and_save_models_test.py"],
  deps=[
    ":load_and_save_models",
  ]
)