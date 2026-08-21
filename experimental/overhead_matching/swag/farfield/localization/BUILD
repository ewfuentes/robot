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

py_library(
    name = "resection",
    srcs = ["resection.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "resection_test",
    srcs = ["resection_test.py"],
    deps = [
        requirement("numpy"),
        ":resection",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "mode_tracker",
    srcs = ["mode_tracker.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":structs",
    ],
)

py_library(
    name = "proposal",
    srcs = ["proposal.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":filter_catalog",
        ":resection",
        ":structs",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        requirement("numpy"),
        ":filter",
        ":metrics",
        ":structs",
    ],
)

py_library(
    name = "filter",
    srcs = ["filter.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
        ":filter_catalog",
        ":metrics",
        ":mode_tracker",
        ":proposal",
        ":structs",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "filter_test",
    srcs = ["filter_test.py"],
    deps = [
        requirement("numpy"),
        ":filter",
        ":filter_catalog",
        ":metrics",
        ":structs",
    ],
)

py_library(
    name = "torch_backend",
    srcs = ["torch_backend.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("torch"),
        ":filter",
        ":filter_catalog",
        ":structs",
        "//common/torch:load_torch_deps",
    ],
)

py_test(
    name = "torch_backend_test",
    srcs = ["torch_backend_test.py"],
    deps = [
        requirement("numpy"),
        requirement("torch"),
        ":filter",
        ":filter_catalog",
        ":structs",
        ":torch_backend",
        "//common/torch:load_torch_deps",
    ],
)

py_library(
    name = "scenario",
    srcs = ["scenario.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        ":filter_catalog",
        ":structs",
        "//common/python:serialization",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "scenario_test",
    srcs = ["scenario_test.py"],
    deps = [
        requirement("numpy"),
        ":scenario",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "mode_tracker_test",
    size = "large",
    srcs = ["mode_tracker_test.py"],
    deps = [
        requirement("numpy"),
        ":filter",
        ":mode_tracker",
        ":scenario",
        ":structs",
    ],
)

py_test(
    name = "proposal_test",
    size = "large",
    srcs = ["proposal_test.py"],
    deps = [
        requirement("numpy"),
        ":filter",
        ":metrics",
        ":proposal",
        ":resection",
        ":scenario",
        ":structs",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "consistency_test",
    size = "large",
    srcs = ["consistency_test.py"],
    deps = [
        requirement("numpy"),
        ":filter",
        ":metrics",
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
        ":filter",
        ":metrics",
        ":scenario",
        ":structs",
    ],
)
