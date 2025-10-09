load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "similarity_engine",
    srcs = ["similarity_engine.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//common/torch:load_torch_deps",
        "//experimental/overhead_matching/swag/model:swag_patch_embedding",
        "//experimental/overhead_matching/swag/model:swag_config_types",
        "//experimental/overhead_matching/swag/model:swag_model_input_output",
        requirement("torch"),
        requirement("torchvision"),
        requirement("pillow"),
        requirement("numpy"),
    ],
)

py_library(
    name = "pca_analyzer",
    srcs = ["pca_analyzer.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("torch"),
        requirement("numpy"),
    ],
)

py_library(
    name = "vigor_browser",
    srcs = ["vigor_browser.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//experimental/overhead_matching/swag/data:vigor_dataset",
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("torch"),
        requirement("pillow"),
    ],
)

py_library(
    name = "landmark_overlay",
    srcs = ["landmark_overlay.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//common/gps:web_mercator",
        requirement("matplotlib"),
        requirement("numpy"),
    ],
)

py_library(
    name = "interactive_viewer",
    srcs = ["interactive_viewer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":similarity_engine",
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pillow"),
        requirement("PyGObject"),
    ],
)

py_library(
    name = "enhanced_interactive_viewer",
    srcs = ["enhanced_interactive_viewer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":similarity_engine",
        ":pca_analyzer",
        ":landmark_overlay",
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pillow"),
    ],
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":interactive_viewer",
        ":enhanced_interactive_viewer",
        ":vigor_browser",
        ":similarity_engine",
        "//experimental/overhead_matching/swag/data:vigor_dataset",
    ],
    main = "main.py",
)

# Convenience alias for easier invocation
py_binary(
    name = "image_similarity_viewer",
    srcs = ["main.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":interactive_viewer",
        ":enhanced_interactive_viewer",
        ":vigor_browser",
        ":similarity_engine",
        "//experimental/overhead_matching/swag/data:vigor_dataset",
    ],
    main = "main.py",
)
