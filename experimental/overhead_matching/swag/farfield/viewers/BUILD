load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "page",
    srcs = ["page.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:provenance",
    ],
)

py_library(
    name = "match_notes",
    srcs = ["match_notes.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:artifact",
    ],
)

py_test(
    name = "match_notes_test",
    srcs = ["match_notes_test.py"],
    deps = [":match_notes"],
)

py_library(
    name = "server_lib",
    srcs = ["server.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("flask"),
        ":match_notes",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield/localization:particle_sampling",
    ],
)

py_binary(
    name = "server",
    srcs = ["server.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":server_lib"],
)

py_test(
    name = "server_test",
    srcs = ["server_test.py"],
    deps = [
        requirement("flask"),
        requirement("numpy"),
        ":match_notes",
        ":server_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield/localization:run_io",
    ],
)

py_library(
    name = "indexes",
    srcs = ["indexes.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":page",
        "//experimental/overhead_matching/swag/farfield:artifact",
    ],
)

py_test(
    name = "indexes_test",
    srcs = ["indexes_test.py"],
    deps = [
        ":indexes",
        ":page",
    ],
)

py_binary(
    name = "refresh_indexes",
    srcs = ["refresh_indexes.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":indexes",
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)
