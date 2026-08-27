load("@pip//:requirements.bzl", "requirement")

# CrossLocate-Depth baseline: RGB -> rendered-DEM-depth retrieval
# (dem_baseline_plan.md / dem_baseline_grounding.md).

py_library(
    name = "terrain",
    srcs = ["terrain.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pyproj"),
        requirement("rasterio"),
    ],
)

py_library(
    name = "lattice",
    srcs = ["lattice.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":terrain",
        requirement("numpy"),
    ],
)

py_library(
    name = "depth_render",
    srcs = ["depth_render.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":terrain",
        requirement("numpy"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
    ],
)

py_library(
    name = "crosslocate_net",
    srcs = ["crosslocate_net.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
    ],
)

py_library(
    name = "query_crops",
    srcs = ["query_crops.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "panorama_score",
    srcs = ["panorama_score.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
    ],
)

py_library(
    name = "render_db_lib",
    srcs = ["render_db.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":crosslocate_net",
        ":depth_render",
        ":lattice",
        ":terrain",
        requirement("numpy"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
    ],
)

py_binary(
    name = "render_db",
    srcs = ["render_db.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":render_db_lib"],
)

py_binary(
    name = "qa_render",
    srcs = ["qa_render.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":depth_render",
        ":terrain",
        requirement("numpy"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
    ],
)

py_binary(
    name = "build_surface",
    srcs = ["build_surface.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":terrain"],
)

py_binary(
    name = "port_verification",
    srcs = ["port_verification.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":crosslocate_net",
        requirement("numpy"),
        requirement("opencv-python"),
        requirement("pillow"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
    ],
)

py_binary(
    name = "framewise_eval",
    srcs = ["framewise_eval.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":crosslocate_net",
        ":panorama_score",
        ":query_crops",
        ":render_db_lib",
        ":terrain",
        requirement("numpy"),
        requirement("pillow"),
        requirement("pyproj"),
        requirement("torch"),
        "//common/torch:load_torch_deps",
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)

# convert_checkpoint.py is intentionally NOT a bazel target: it needs
# TensorFlow, which stays out of this repo's dependency set. Run it with
# `uv run --with tensorflow-cpu` per its docstring.

py_test(
    name = "terrain_test",
    srcs = ["terrain_test.py"],
    deps = [
        ":lattice",
        ":terrain",
    ],
)

py_test(
    name = "depth_render_test",
    srcs = ["depth_render_test.py"],
    deps = [
        ":depth_render",
        ":terrain",
    ],
)

py_test(
    name = "crosslocate_net_test",
    srcs = ["crosslocate_net_test.py"],
    deps = [":crosslocate_net"],
)

py_test(
    name = "query_crops_test",
    srcs = ["query_crops_test.py"],
    deps = [
        ":query_crops",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "panorama_score_test",
    srcs = ["panorama_score_test.py"],
    deps = [":panorama_score"],
)
