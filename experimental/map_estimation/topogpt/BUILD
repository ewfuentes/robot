load("@pip//:requirements.bzl", "requirement")

package(default_visibility = ["//experimental/map_estimation:__subpackages__"])

py_test(
    name = "map_prior_smoke_test",
    size = "small",
    srcs = ["map_prior_smoke_test.py"],
    deps = [
        # The fork declares MapPrior's dependencies, so requirement("MapPrior") carries its
        # import closure -- cv2, shapely, scikit-learn, hydra and the rest -- transitively.
        # Named here are only what this file imports itself. load_torch_deps supplies torch;
        # MapPrior's dataset-generation extras are not installed and must not be imported.
        "//common/torch:load_torch_deps",
        requirement("MapPrior"),
        requirement("lightning"),
        requirement("omegaconf"),
    ],
)

py_library(
    name = "run_prior_lib",
    srcs = ["run_prior.py"],
    deps = [
        "//common/torch:load_torch_deps",
        "//experimental/map_estimation/data:argoverse_layout",
        "//experimental/map_estimation/viz:av2_source",
        requirement("MapPrior"),
        requirement("numpy"),
        requirement("omegaconf"),
    ],
)

py_binary(
    name = "run_prior",
    srcs = ["run_prior.py"],
    main = "run_prior.py",
    deps = [":run_prior_lib"],
)

py_binary(
    name = "run_finetune",
    srcs = ["run_finetune.py"],
    deps = [
        ":run_prior_lib",
        "//common/torch:load_torch_deps",
        "//experimental/map_estimation/data:argoverse_layout",
        "//experimental/map_estimation/viz:av2_scene",
        "//experimental/map_estimation/viz:av2_source",
        "//experimental/map_estimation/viz:view_log",
        requirement("MapPrior"),
        requirement("numpy"),
        requirement("omegaconf"),
        requirement("rerun-sdk"),
        # modules/flow is reachable only as a hydra _target_, so nothing else pulls it in.
        requirement("torchcfm"),
    ],
)
