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
    name = "resection",
    srcs = ["resection.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":geodesy",
    ],
)

py_test(
    name = "resection_test",
    srcs = ["resection_test.py"],
    deps = [
        requirement("numpy"),
        ":geodesy",
        ":resection",
    ],
)

py_library(
    name = "proposal",
    srcs = ["proposal.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":catalog",
        ":geodesy",
        ":resection",
        ":structs",
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
    name = "filter",
    srcs = ["filter.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
        ":catalog",
        ":geodesy",
        ":mode_tracker",
        ":proposal",
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
    name = "torch_backend",
    srcs = ["torch_backend.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
        ":catalog",
        ":filter",
        ":structs",
    ],
)

py_test(
    name = "torch_backend_test",
    srcs = ["torch_backend_test.py"],
    deps = [
        requirement("numpy"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
        ":catalog",
        ":filter",
        ":structs",
        ":torch_backend",
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

py_test(
    name = "proposal_test",
    size = "large",
    srcs = ["proposal_test.py"],
    deps = [
        requirement("numpy"),
        ":filter",
        ":proposal",
        ":resection",
        ":scenario",
        ":structs",
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
        ":proposal",
        ":scenario",
        ":structs",
    ],
)


py_binary(
    name = "viewer",
    srcs = ["viewer.py"],
    deps = [
        requirement("numpy"),
        ":filter",
        ":geodesy",
        ":run_log",
    ],
)

py_library(
    name = "export_ingest",
    srcs = ["export_ingest.py"],
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

py_binary(
    name = "run_export",
    srcs = ["run_export.py"],
    deps = [
        requirement("numpy"),
        ":export_ingest",
        ":filter",
        ":geodesy",
        ":run_log",
        ":structs",
        ":torch_backend",
    ],
)

py_binary(
    name = "m4_demo_runs",
    srcs = ["m4_demo_runs.py"],
    deps = [
        requirement("numpy"),
        ":filter",
        ":run_log",
        ":scenario",
        ":structs",
    ],
)

py_library(
    name = "gps_to_odometry_lib",
    srcs = ["gps_to_odometry.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        "//common/python:serialization",
        ":geodesy",
        ":structs",
    ],
)

py_binary(
    name = "gps_to_odometry",
    srcs = ["gps_to_odometry.py"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        "//common/python:serialization",
        ":geodesy",
        ":structs",
    ],
)

py_test(
    name = "gps_to_odometry_test",
    srcs = ["gps_to_odometry_test.py"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        "//common/python:serialization",
        ":gps_to_odometry_lib",
        ":structs",
    ],
)

