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
