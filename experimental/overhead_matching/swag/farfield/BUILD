load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
    ],
)

py_test(
    name = "geometry_test",
    srcs = ["geometry_test.py"],
    deps = [
        requirement("numpy"),
        ":geometry",
    ],
)
