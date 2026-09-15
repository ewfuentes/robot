load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "schema",
    srcs = ["schema.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("geopandas"),
        requirement("pandas"),
        requirement("pyarrow"),
    ],
)

py_test(
    name = "schema_test",
    srcs = ["schema_test.py"],
    deps = [
        requirement("geopandas"),
        requirement("shapely"),
        ":schema",
    ],
)

py_library(
    name = "catalog",
    srcs = ["catalog.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("shapely"),
        ":schema",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "lineage",
    srcs = ["lineage.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":schema",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)

py_test(
    name = "lineage_test",
    srcs = ["lineage_test.py"],
    deps = [
        ":lineage",
        ":schema",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)

py_test(
    name = "catalog_test",
    srcs = ["catalog_test.py"],
    deps = [
        requirement("numpy"),
        requirement("shapely"),
        ":catalog",
        ":schema",
    ],
)
