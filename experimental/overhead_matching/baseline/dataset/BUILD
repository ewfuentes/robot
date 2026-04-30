load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "tile_geometry",
    srcs = ["tile_geometry.py"],
    visibility = ["//experimental/overhead_matching/baseline:__subpackages__"],
    deps = [
        requirement("pandas"),
        "//common/gps:web_mercator",
    ],
)

py_test(
    name = "tile_geometry_test",
    srcs = ["tile_geometry_test.py"],
    deps = [":tile_geometry"],
)

py_library(
    name = "city_pbf_map",
    srcs = ["city_pbf_map.py"],
    visibility = ["//experimental/overhead_matching/baseline:__subpackages__"],
)

py_test(
    name = "city_pbf_map_test",
    srcs = ["city_pbf_map_test.py"],
    deps = [":city_pbf_map"],
)

py_library(
    name = "style_template",
    srcs = ["style_template.py"],
    visibility = ["//experimental/overhead_matching/baseline:__subpackages__"],
    data = [
        "@osm_bright_gl_style//:files",
        "@openmaptiles_fonts//:files",
    ],
    deps = ["@rules_python//python/runfiles"],
)

py_test(
    name = "style_template_test",
    srcs = ["style_template_test.py"],
    deps = [":style_template"],
)

py_binary(
    name = "bake_mbtiles",
    srcs = ["bake_mbtiles.py"],
    data = ["@planetiler_jar//file"],
    deps = [
        ":city_pbf_map",
        ":tile_geometry",
        "@rules_python//python/runfiles",
    ],
)

py_binary(
    name = "render_osm_tiles",
    srcs = ["render_osm_tiles.py"],
    deps = [
        ":city_pbf_map",
        ":style_template",
        ":tile_geometry",
        requirement("pymgl"),
        requirement("tqdm"),
    ],
)

py_library(
    name = "osm_org_tile",
    srcs = ["osm_org_tile.py"],
    visibility = ["//experimental/overhead_matching/baseline:__subpackages__"],
    deps = [
        ":tile_geometry",
        "//common/gps:web_mercator",
        requirement("requests"),
        requirement("Pillow"),
    ],
)

py_binary(
    name = "compare_tiles",
    srcs = ["compare_tiles.py"],
    deps = [
        ":city_pbf_map",
        ":osm_org_tile",
        ":tile_geometry",
        requirement("Pillow"),
    ],
)

