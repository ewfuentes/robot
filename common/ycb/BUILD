load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "downloader",
    srcs = ["downloader.py"],
    deps = [
      requirement("requests"),
      requirement("beautifulsoup4"),
    ]
)
