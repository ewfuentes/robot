load("@pip//:requirements.bzl", "requirement")

exports_files(glob(["configs/*.yaml"]))

py_library(
    name = "region_lib",
    srcs = ["region.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:artifact_identity",
        "//experimental/overhead_matching/swag/farfield:artifact_recipe",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
        "//experimental/overhead_matching/swag/farfield/catalog:schema",
        "//experimental/overhead_matching/swag/farfield/collection:active_catalogs_lib",
    ],
)

py_binary(
    name = "region",
    srcs = ["region.py"],
    deps = [":region_lib"],
)

py_test(
    name = "input_smoke_test",
    srcs = [
        "input_smoke.py",
        "input_smoke_test.py",
    ],
    main = "input_smoke_test.py",
    deps = [
        ":region_lib",
        "//experimental/overhead_matching/swag/data:vigor_dataset",
        "//experimental/overhead_matching/swag/farfield:artifact",
    ],
)

py_test(
    name = "region_test",
    srcs = ["region_test.py"],
    deps = [":region_lib"],
)

py_library(
    name = "satellite_lib",
    srcs = ["satellite.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":region_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:artifact_identity",
        "//experimental/overhead_matching/swag/farfield:artifact_recipe",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
        requirement("pillow"),
        requirement("pyproj"),
        requirement("rasterio"),
        requirement("requests"),
        requirement("shapely"),
    ],
)

py_binary(
    name = "satellite",
    srcs = ["satellite.py"],
    deps = [":satellite_lib"],
)

py_test(
    name = "satellite_test",
    srcs = ["satellite_test.py"],
    deps = [
        ":satellite_lib",
        requirement("pillow"),
    ],
)

py_library(
    name = "osm_lib",
    srcs = ["osm.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("shapely"),
        ":region_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:artifact_identity",
        "//experimental/overhead_matching/swag/farfield:artifact_recipe",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
        "//experimental/overhead_matching/swag/farfield/catalog:schema",
        "//experimental/overhead_matching/swag/model:semantic_landmark_utils",
    ],
)

py_binary(
    name = "osm",
    srcs = ["osm.py"],
    deps = [":osm_lib"],
)

py_test(
    name = "osm_test",
    srcs = ["osm_test.py"],
    deps = [
        requirement("shapely"),
        ":osm_lib",
        ":region_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield/catalog:schema",
    ],
)

py_binary(
    name = "input_smoke",
    srcs = ["input_smoke.py"],
    deps = [
        ":region_lib",
        "//experimental/overhead_matching/swag/data:vigor_dataset",
        "//experimental/overhead_matching/swag/farfield:artifact",
    ],
)

py_binary(
    name = "vlm_requests",
    srcs = ["vlm_requests.py"],
    deps = [requirement("pillow")],
)

py_library(
    name = "publish_staged_lib",
    srcs = ["publish_staged.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:code_provenance",
    ],
)

py_binary(
    name = "publish_staged",
    srcs = ["publish_staged.py"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:code_provenance",
    ],
)

py_test(
    name = "publish_staged_test",
    srcs = ["publish_staged_test.py"],
    deps = [
        ":publish_staged_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
    ],
)
