load("@pip//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_binary")

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

cc_library(
  name = "test_function",
  hdrs = ["test_function.hh"],
  srcs = ["test_function.cc"],
)

pybind_extension(
  name = "pybind_example_python",
  srcs = ["pybind_example_python.cc"],
  copts = ["-fPIC"],
  deps = [
    ":test_function",
  ]
)

py_test(
  name = "pybind_example_test",
  srcs = ["pybind_example_test.py"],
  data = [":pybind_example_python.so"],
)
