load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "geodesy",
    srcs = ["geodesy.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        "//experimental/overhead_matching/swag/landmark_filtering:bearing_geometry",
    ],
)

py_test(
    name = "geodesy_test",
    srcs = ["geodesy_test.py"],
    deps = [
        requirement("numpy"),
        "//experimental/overhead_matching/swag/landmark_filtering:bearing_geometry",
        ":geodesy",
    ],
)

py_library(
    name = "structs",
    srcs = ["structs.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        "//common/python:serialization",
    ],
)

py_library(
    name = "catalog",
    srcs = ["catalog.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":geodesy",
    ],
)

py_library(
    name = "filter",
    srcs = ["filter.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
        ":catalog",
        ":geodesy",
        ":structs",
    ],
)

py_test(
    name = "filter_test",
    srcs = ["filter_test.py"],
    deps = [
        requirement("numpy"),
        ":catalog",
        ":filter",
        ":structs",
    ],
)

py_library(
    name = "scenario",
    srcs = ["scenario.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        "//common/python:serialization",
        ":catalog",
        ":geodesy",
        ":structs",
    ],
)

py_test(
    name = "scenario_test",
    srcs = ["scenario_test.py"],
    deps = [
        requirement("numpy"),
        ":catalog",
        ":geodesy",
        ":scenario",
    ],
)

py_library(
    name = "run_log",
    srcs = ["run_log.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        "//common/python:serialization",
        ":structs",
    ],
)

py_test(
    name = "run_log_test",
    srcs = ["run_log_test.py"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        ":catalog",
        ":filter",
        ":run_log",
        ":scenario",
        ":structs",
    ],
)

py_test(
    name = "e2e_convergence_test",
    size = "large",
    srcs = ["e2e_convergence_test.py"],
    deps = [
        requirement("numpy"),
        ":catalog",
        ":filter",
        ":scenario",
        ":structs",
    ],
)

py_binary(
    name = "run_localization",
    srcs = ["run_localization.py"],
    deps = [
        requirement("numpy"),
        ":catalog",
        ":filter",
        ":run_log",
        ":scenario",
        ":structs",
    ],
)

py_binary(
    name = "plot_run",
    srcs = ["plot_run.py"],
    deps = [
        requirement("matplotlib"),
        requirement("numpy"),
        ":catalog",
        ":filter",
        ":geodesy",
        ":run_log",
    ],
)



py_test(
    name = "consistency_test",
    size = "large",
    srcs = ["consistency_test.py"],
    deps = [
        requirement("numpy"),
        ":filter",
        ":scenario",
        ":structs",
    ],
)
