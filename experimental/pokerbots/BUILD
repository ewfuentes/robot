load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_runtime")
load("@rules_pkg//pkg:zip.bzl", "pkg_zip")

pkg_zip(
    name="zipped_bot",
    srcs=[":pokerbot_zip", "commands.json"],
)

filegroup(
    name="pokerbot_zip",
    srcs=[
        ":pokerbot",
    ],
    output_group="python_zip_file",
)

py_runtime(
    name="system_runtime",
    python_version="PY3",
    interpreter_path="/usr/bin/python3",
)

pybind_extension(
  name = "hand_evaluator_python",
  srcs = ["hand_evaluator_python.cc"],
  copts = ["-fPIC"],
  deps = [
    "@ompeval//:ompeval_lib",
  ]
)

py_test(
  name = "hand_evaluator_python_test",
  srcs = ["hand_evaluator_python_test.py"],
  data = [":hand_evaluator_python.so"],
)

py_binary(
    name="pokerbot",
    srcs=["pokerbot.py"],
    data = [
      "//common/python:pybind_example_python.so",
      ":hand_evaluator_python.so",
    ],
    deps=[
        "@mit_pokerbots//:framework",
    ],
)

cc_binary(
  name = "pokerbot_master",
  srcs = ["pokerbot_master.cc"],
  deps = [
    "//domain:rob_poker",
    "//learning:cfr",
    "@cxxopts//:cxxopts",
  ]
)