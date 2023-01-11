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
    interpreter_path=select(
        {
            # Update paths as appropriate for your system.
            "@bazel_tools//tools/python:PY2": "/usr/bin/python",
            "@bazel_tools//tools/python:PY3": "/usr/bin/python3",
        }
    ),
    files=[],
)

py_binary(
    name="pokerbot",
    srcs=["pokerbot.py"],
    deps=[
        "@mit_pokerbots//:framework",
    ],
)
