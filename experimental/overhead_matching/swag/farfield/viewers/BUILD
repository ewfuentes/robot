load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "page",
    srcs = ["page.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_library(
    name = "indexes",
    srcs = ["indexes.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":page",
        "//experimental/overhead_matching/swag/farfield:artifact",
    ],
)

py_test(
    name = "indexes_test",
    srcs = ["indexes_test.py"],
    deps = [
        ":indexes",
        ":page",
    ],
)

py_binary(
    name = "refresh_indexes",
    srcs = ["refresh_indexes.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":indexes",
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)
