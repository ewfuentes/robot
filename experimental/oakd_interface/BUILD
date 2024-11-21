
load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "oakd_interface",
    srcs = ["oakd_interface.py"],
    deps = [
	requirement("depthai"),
	requirement("numpy"),
    ]
)


