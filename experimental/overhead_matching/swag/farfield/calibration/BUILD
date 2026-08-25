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
    deps = [
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield/tracking:semantic_audit",
    ],
)

py_test(
    name = "audit_io_test",
    srcs = ["audit_io_test.py"],
    deps = [
        ":audit_io",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)

py_library(
    name = "mount_offset_sweep_lib",
    srcs = ["mount_offset_sweep.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        "//experimental/overhead_matching/swag/farfield:geometry",
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
        "//experimental/overhead_matching/swag/farfield:geometry",
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

py_library(
    name = "build_alignment_diagnostics_lib",
    srcs = ["build_alignment_diagnostics.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":heading",
        ":mount_offset_sweep_lib",
        ":sun_offset_check_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
    ],
)

py_binary(
    name = "build_alignment_diagnostics",
    srcs = ["build_alignment_diagnostics.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":heading",
        ":mount_offset_sweep_lib",
        ":sun_offset_check_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
    ],
)

py_test(
    name = "build_alignment_diagnostics_test",
    srcs = ["build_alignment_diagnostics_test.py"],
    deps = [
        requirement("pillow"),
        ":build_alignment_diagnostics_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)
