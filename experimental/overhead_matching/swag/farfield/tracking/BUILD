load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "tracklets",
    srcs = ["tracklets.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "tracklets_test",
    srcs = ["tracklets_test.py"],
    deps = [
        requirement("numpy"),
        ":tracklets",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)
