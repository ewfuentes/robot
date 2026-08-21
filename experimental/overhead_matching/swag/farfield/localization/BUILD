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

py_library(
    name = "build_export_lib",
    srcs = ["build_export.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        requirement("pillow"),
        ":export_ingest",
        ":gps_to_odometry_lib",
        ":run_io",
        ":structs",
        "//common/python:serialization",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
        "//experimental/overhead_matching/swag/farfield/calibration:heading",
        "//experimental/overhead_matching/swag/farfield/catalog",
        "//experimental/overhead_matching/swag/farfield/tracking:tracklets",
    ],
)

py_binary(
    name = "build_export",
    srcs = ["build_export.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":build_export_lib"],
)

py_test(
    name = "build_export_test",
    srcs = ["build_export_test.py"],
    deps = [
        requirement("msgspec"),
        requirement("shapely"),
        ":build_export_lib",
        ":export_ingest",
        ":structs",
        "//common/python:serialization",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:testing",
        "//experimental/overhead_matching/swag/farfield/catalog:schema",
        "//experimental/overhead_matching/swag/farfield/tracking:tracklets",
    ],
)

py_library(
    name = "run_export_lib",
    srcs = ["run_export.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":export_ingest",
        ":filter",
        ":metrics",
        ":run_io",
        ":structs",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_binary(
    name = "run_export",
    srcs = ["run_export.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":run_export_lib"],
)

py_library(
    name = "run_localization_lib",
    srcs = ["run_localization.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        ":filter",
        ":metrics",
        ":run_io",
        ":scenario",
        ":structs",
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_binary(
    name = "run_localization",
    srcs = ["run_localization.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":run_localization_lib"],
)

py_test(
    name = "run_drivers_test",
    srcs = ["run_drivers_test.py"],
    deps = [
        requirement("numpy"),
        ":run_export_lib",
        ":run_io",
        ":run_localization_lib",
        ":structs",
    ],
)

py_library(
    name = "sources",
    srcs = ["sources.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_library(
    name = "forensics_lib",
    srcs = ["forensics.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "forensics_test",
    srcs = ["forensics_test.py"],
    deps = [
        requirement("numpy"),
        ":filter_catalog",
        ":forensics_lib",
        ":run_io",
        ":sources",
        ":structs",
    ],
)

py_library(
    name = "replay",
    srcs = ["replay.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        requirement("numpy"),
        ":filter",
        ":filter_catalog",
        ":run_io",
        ":structs",
        "//common/python:serialization",
        "//experimental/overhead_matching/swag/farfield:geometry",
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
        ":filter",
        ":replay",
        "//common/python:serialization",
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
        ":run_io",
        ":scenario",
        ":structs",
    ],
)

py_library(
    name = "basemap_lib",
    srcs = ["basemap.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield/catalog:schema",
    ],
)

py_binary(
    name = "basemap",
    srcs = ["basemap.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":basemap_lib"],
)

py_library(
    name = "satellite_underlay_lib",
    srcs = ["satellite_underlay.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        requirement("requests"),
        ":run_io",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_binary(
    name = "satellite_underlay",
    srcs = ["satellite_underlay.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":satellite_underlay_lib"],
)

py_test(
    name = "satellite_underlay_test",
    srcs = ["satellite_underlay_test.py"],
    deps = [
        ":satellite_underlay_lib",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_binary(
    name = "plot_run",
    srcs = ["plots.py"],
    main = "plots.py",
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("matplotlib"),
        requirement("numpy"),
        ":metrics",
        ":run_io",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:provenance",
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
        ":forensics_lib",
        ":metrics",
        ":replay",
        ":run_io",
        ":sources",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "viewer_lib",
    srcs = ["viewer.py"],
    data = [
        "viewer_assets/app.js",
        "viewer_assets/style.css",
    ],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":viewer_payload",
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_binary(
    name = "viewer",
    srcs = ["viewer.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":viewer_lib"],
)

py_library(
    name = "viewer_server_lib",
    srcs = ["viewer_server.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("flask"),
        requirement("msgspec"),
        requirement("numpy"),
        ":replay",
        ":run_io",
        ":sources",
        ":viewer_lib",
        ":viewer_payload",
    ],
)

py_binary(
    name = "viewer_server",
    srcs = ["viewer_server.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        # torch_backend so a run recorded with the GPU backend can serve
        # live counterfactual replays (filter imports it lazily by config).
        requirement("torch"),
        ":torch_backend",
        ":viewer_server_lib",
        "//common/torch:load_torch_deps",
    ],
)

py_test(
    name = "viewer_test",
    size = "large",
    srcs = ["viewer_test.py"],
    deps = [
        requirement("flask"),
        requirement("msgspec"),
        requirement("numpy"),
        ":attribution",
        ":filter",
        ":run_io",
        ":scenario",
        ":structs",
        ":viewer_lib",
        ":viewer_payload",
        ":viewer_server_lib",
    ],
)

py_binary(
    name = "forensics",
    srcs = ["forensics_cli.py"],
    main = "forensics_cli.py",
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        # torch_backend so `forensics replay/attribute` can replay runs
        # recorded with measurement_backend="torch".
        requirement("torch"),
        ":attribution",
        ":forensics_lib",
        ":metrics",
        ":replay",
        ":run_io",
        ":sources",
        ":torch_backend",
        "//common/torch:load_torch_deps",
    ],
)
