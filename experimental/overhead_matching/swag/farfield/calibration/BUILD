load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "heading",
    srcs = ["heading.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_test(
    name = "heading_test",
    srcs = ["heading_test.py"],
    deps = [
        requirement("numpy"),
        ":heading",
    ],
)

py_library(
    name = "audit_io",
    srcs = ["audit_io.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_test(
    name = "audit_io_test",
    srcs = ["audit_io_test.py"],
    deps = [":audit_io"],
)

py_library(
    name = "mount_offset_sweep_lib",
    srcs = ["mount_offset_sweep.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":audit_io",
        ":heading",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield/tracking:tracklets",
    ],
)

py_binary(
    name = "mount_offset_sweep",
    srcs = ["mount_offset_sweep.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":audit_io",
        ":heading",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield/tracking:tracklets",
    ],
)

py_test(
    name = "mount_offset_sweep_test",
    srcs = ["mount_offset_sweep_test.py"],
    deps = [
        ":mount_offset_sweep_lib",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)

py_library(
    name = "sun_offset_check_lib",
    srcs = ["sun_offset_check.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":heading",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_binary(
    name = "sun_offset_check",
    srcs = ["sun_offset_check.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":heading",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_test(
    name = "sun_offset_check_test",
    srcs = ["sun_offset_check_test.py"],
    deps = [
        requirement("numpy"),
        ":sun_offset_check_lib",
        "//experimental/overhead_matching/swag/farfield:geometry",
    ],
)
