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
    files=[],
)

py_binary(
    name="pokerbot",
    srcs=["pokerbot.py"],
    data = [
      "//common/python:pybind_example_python.so"
    ],
    deps=[
        "@mit_pokerbots//:framework",
    ],
)
