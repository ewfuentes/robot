load("@pip//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
  name = "jupyter_notebook",
  srcs = ["jupyter_notebook.py"],
  data = [
    "//experimental/beacon_sim:correlated_beacons_python.so",
  ],
  deps = [
    requirement("notebook"),
    requirement("matplotlib"),
    requirement("Pillow"),
    requirement("numpy"),
    requirement("pandas"),
    requirement("scikit-learn"),
    requirement("tqdm"),
    "//experimental/beacon_sim:environment_builder",
    "//experimental/beacon_sim:plot_trials",
  ],
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
  visibility=["//visibility:public"],
  deps = [
    ":test_function",
  ]
)

py_test(
  name = "pybind_example_test",
  srcs = ["pybind_example_test.py"],
  data = [":pybind_example_python.so"],
)
