
package(features=["warning_compile_flags"])

load("@pip//:requirements.bzl", "requirement", "data_requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_binary")
load("@python//3.8:defs.bzl", py_test_3_8 = "py_test")

py_binary(
  name = "marimo_server",
  srcs = ["marimo_server.py"],
  data = ["//:.marimo.toml"],
  deps = [
    requirement("marimo"),
    requirement("numpy"),
    requirement("altair"),
    requirement("pandas"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/learned/model:clevr_transformer",
    "//experimental/overhead_matching/learned/model:clevr_tokenizer",
    "//experimental/overhead_matching/learned/data:clevr_dataset",
    "//experimental/overhead_matching/learned/scripts:train_clevr_transformer",
    "//experimental/overhead_matching/swag/evaluation:evaluate_swag",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/model:patch_embedding",
  ]
)

py_binary(
  name = "jupyter_notebook",
  srcs = ["jupyter_notebook.py"],
  data = [
    "//experimental/beacon_sim:correlated_beacons_python.so",
    "//experimental/beacon_sim:ekf_slam_python.so",
    "//experimental/beacon_sim:belief_road_map_planner_python.so",
    "@embag//:embag_python.so",
    "//common/liegroups:se2_python.so",
    "//common/time:robot_time_python.so",
    "//planning:probabilistic_road_map_python.so",
    data_requirement("jupyterlab"),
  ],
  deps = [
    requirement("jupyterlab"),
    requirement("ipympl"),
    requirement("matplotlib"),
    requirement("Pillow"),
    requirement("numpy"),
    requirement("pandas"),
    requirement("scipy"),
    requirement("scikit-learn"),
    requirement("opencv-python"),
    requirement("supervision"),
    requirement("tqdm"),
    requirement("seaborn"),
    requirement("reportlab"),
    "//experimental/overhead_matching:grounding_sam",
    "//experimental/beacon_sim:environment_builder",
    "//experimental/beacon_sim:plot_trials",
    "//experimental/beacon_sim:experiment_results_proto_py",
    "//common/torch:load_torch_deps",
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

py_test_3_8(
  name = "pybind_example_3_8_test",
  srcs = ["pybind_example_test.py"],
  main = "pybind_example_test.py",
  data = [":pybind_example_python.so"],
)

py_test(
  name = "hermeticity_test",
  srcs = ["hermeticity_test.py"],
)
