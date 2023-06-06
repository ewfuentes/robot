load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

py_binary(
    name = "downloader",
    srcs = ["downloader.py"],
    deps = [
      requirement("requests"),
      requirement("beautifulsoup4"),
    ]
)

py_test(
    name = "load_ycb_test",
    srcs = ["load_ycb_test.py"],
    tags=["manual"],
    deps = [
        requirement("IPython"),
        "@drake//bindings/pydrake",
    ]
)
