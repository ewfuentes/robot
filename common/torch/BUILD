
load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

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
    requirement("dinov2"),
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

cuda_library(
  name = "example_kernel",
  hdrs = ["example_kernel.hh"],
  srcs = ["example_kernel.cu"],
  deps = [
    "@libtorch//:headers",
  ]
)

pybind_extension(
  name = "example_kernel_python",
  srcs = ["example_kernel_python.cc"],
  deps = [":example_kernel"],
)

py_test(
  name = "example_kernel_test",
  srcs = ["example_kernel_test.py"],
  data = [":example_kernel_python.so"],
  deps = [
    ":load_torch_deps",
    requirement("numpy"),
  ]
)
