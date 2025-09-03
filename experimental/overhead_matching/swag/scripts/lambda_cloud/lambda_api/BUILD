load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip//:requirements.bzl", "requirement")
py_library(
    name = "lambda_cloud",
    srcs = [
        "__init__.py",
        "client.py",
        "models.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        requirement("requests"),
        requirement("urllib3"),
    ],
)

py_binary(
    name = "cli",
    srcs = ["cli.py"],
    main = "cli.py",
    visibility = ["//visibility:public"],
    deps = [":lambda_cloud"],
)
