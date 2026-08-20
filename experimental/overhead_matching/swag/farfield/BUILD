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

py_library(
    name = "paths",
    srcs = ["paths.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_test(
    name = "paths_test",
    srcs = ["paths_test.py"],
    deps = [":paths"],
)

py_library(
    name = "provenance",
    srcs = ["provenance.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_test(
    name = "provenance_test",
    srcs = ["provenance_test.py"],
    deps = [":provenance"],
)

py_library(
    name = "run_config",
    srcs = ["run_config.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":provenance"],
)

py_test(
    name = "run_config_test",
    srcs = ["run_config_test.py"],
    deps = [":run_config"],
)
