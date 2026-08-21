load("@pip//:requirements.bzl", "requirement")

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
    name = "filter_catalog",
    srcs = ["filter_catalog.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "filter_catalog_test",
    srcs = ["filter_catalog_test.py"],
    deps = [
        requirement("numpy"),
        ":filter_catalog",
    ],
)

py_library(
    name = "run_io",
    srcs = ["run_io.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        ":structs",
        "//common/python:serialization",
    ],
)

py_test(
    name = "run_io_test",
    srcs = ["run_io_test.py"],
    deps = [
        requirement("numpy"),
        ":run_io",
        ":structs",
    ],
)

py_library(
    name = "export_ingest",
    srcs = ["export_ingest.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        ":filter_catalog",
        ":run_io",
        ":structs",
        "//common/python:serialization",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "export_ingest_test",
    srcs = ["export_ingest_test.py"],
    deps = [
        requirement("msgspec"),
        ":export_ingest",
        ":run_io",
        ":structs",
        "//common/python:serialization",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "gps_to_odometry_lib",
    srcs = ["gps_to_odometry.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        ":structs",
        "//common/python:serialization",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_binary(
    name = "gps_to_odometry",
    srcs = ["gps_to_odometry.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":gps_to_odometry_lib"],
)

py_test(
    name = "gps_to_odometry_test",
    srcs = ["gps_to_odometry_test.py"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        ":gps_to_odometry_lib",
        ":structs",
        "//common/python:serialization",
    ],
)
