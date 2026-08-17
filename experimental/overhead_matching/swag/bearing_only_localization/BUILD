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
    name = "viewer_server",
    srcs = ["viewer_server.py"],
    deps = [
        requirement("flask"),
        requirement("msgspec"),
        requirement("numpy"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
        ":replay",
        ":run_log",
        ":sources",
        ":torch_backend",
        ":viewer_lib",
        ":viewer_payload",
    ],
)

py_test(
    name = "viewer_test",
    size = "large",
    srcs = ["viewer_test.py"],
    deps = [
        requirement("flask"),
        requirement("numpy"),
        ":attribution",
        ":filter",
        ":run_log",
        ":scenario",
        ":structs",
        ":viewer_lib",
        ":viewer_payload",
        ":viewer_server",
    ],
)

py_binary(
    name = "runlog",
    srcs = ["runlog.py"],
    deps = [
        requirement("numpy"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
        ":attribution",
        ":forensics",
        ":replay",
        ":run_log",
        ":sources",
        ":torch_backend",
    ],
)

py_library(
    name = "viewer_payload",
    srcs = ["viewer_payload.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":attribution",
        ":basemap_lib",
        ":forensics",
        ":geodesy",
        ":replay",
        ":run_log",
        ":sources",
    ],
)

py_library(
    name = "viewer_lib",
    srcs = ["viewer.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":viewer_payload",
    ],
)

py_binary(
    name = "viewer",
    srcs = ["viewer.py"],
    deps = [
        requirement("numpy"),
        ":viewer_payload",
    ],
)

py_library(
    name = "replay",
    srcs = ["replay.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        "//common/python:serialization",
        ":catalog",
        ":filter",
        ":geodesy",
        ":run_log",
        ":structs",
    ],
)

py_library(
    name = "attribution",
    srcs = ["attribution.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        requirement("scipy"),
        "//common/python:serialization",
        ":filter",
        ":replay",
    ],
)

py_library(
    name = "forensics",
    srcs = ["forensics.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":geodesy",
    ],
)

py_library(
    name = "sources",
    srcs = ["sources.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_library(
    name = "basemap_lib",
    srcs = ["basemap.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("geopandas"),
        requirement("numpy"),
        requirement("pyarrow"),
        "//experimental/overhead_matching/swag/data:landmark_schema",
        ":geodesy",
    ],
)

py_binary(
    name = "basemap",
    srcs = ["basemap.py"],
    deps = [
        requirement("geopandas"),
        requirement("numpy"),
        requirement("pyarrow"),
        "//experimental/overhead_matching/swag/data:landmark_schema",
        ":geodesy",
    ],
)

py_test(
    name = "forensics_test",
    srcs = ["forensics_test.py"],
    deps = [
        requirement("numpy"),
        ":catalog",
        ":filter",
        ":forensics",
        ":run_log",
        ":scenario",
        ":sources",
        ":structs",
    ],
)

py_test(
    name = "replay_test",
    size = "large",
    srcs = ["replay_test.py"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        ":attribution",
        ":filter",
        ":replay",
        ":run_log",
        ":scenario",
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

