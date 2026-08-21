load("@pip//:requirements.bzl", "requirement")

package(default_visibility = ["//experimental/map_estimation:__subpackages__"])

py_library(
    name = "s5cmd",
    srcs = ["s5cmd.py"],
    deps = [requirement("msgspec")],
)

py_test(
    name = "s5cmd_test",
    size = "small",
    srcs = ["s5cmd_test.py"],
    deps = [":s5cmd"],
)

py_library(
    name = "argoverse_layout",
    srcs = ["argoverse_layout.py"],
    deps = [
        "//common/python:serialization",
        requirement("msgspec"),
    ],
)

py_test(
    name = "argoverse_layout_test",
    size = "small",
    srcs = ["argoverse_layout_test.py"],
    deps = [":argoverse_layout"],
)

py_library(
    name = "av2_log",
    srcs = ["av2_log.py"],
    deps = [
        ":argoverse_layout",
        requirement("av2"),
    ],
)

py_test(
    name = "av2_log_test",
    size = "small",
    srcs = ["av2_log_test.py"],
    deps = [":av2_log"],
)

py_library(
    name = "argoverse_catalog",
    srcs = ["argoverse_catalog.py"],
    deps = [
        ":argoverse_layout",
        ":s5cmd",
        "//common/python:serialization",
        requirement("msgspec"),
    ],
)

py_test(
    name = "argoverse_catalog_test",
    size = "small",
    srcs = ["argoverse_catalog_test.py"],
    deps = [":argoverse_catalog"],
)

py_library(
    name = "argoverse_download",
    srcs = ["argoverse_download.py"],
    deps = [
        ":argoverse_catalog",
        ":argoverse_layout",
        ":s5cmd",
        "//common/python:serialization",
        requirement("msgspec"),
    ],
)

py_test(
    name = "argoverse_download_test",
    size = "small",
    srcs = ["argoverse_download_test.py"],
    deps = [":argoverse_download"],
)

py_library(
    name = "argoverse_cli_lib",
    srcs = ["argoverse_cli.py"],
    deps = [
        ":argoverse_catalog",
        ":argoverse_download",
        ":argoverse_layout",
        ":s5cmd",
        requirement("msgspec"),
    ],
)

py_binary(
    name = "argoverse",
    srcs = ["argoverse_cli.py"],
    main = "argoverse_cli.py",
    visibility = ["//visibility:public"],
    deps = [":argoverse_cli_lib"],
)

py_test(
    name = "argoverse_cli_test",
    size = "small",
    srcs = ["argoverse_cli_test.py"],
    deps = [":argoverse_cli_lib"],
)

# The download half of this package does not use av2 -- only av2_log and the datasets over it
# do -- so this guards the opencv pin av2 requires independently of any of them.
py_test(
    name = "av2_smoke_test",
    size = "small",
    srcs = ["av2_smoke_test.py"],
    deps = [requirement("av2")],
)
