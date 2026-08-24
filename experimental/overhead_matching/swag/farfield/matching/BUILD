load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "match_landmarks_lib",
    srcs = ["match_landmarks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        "//common/python:serialization",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
        "//experimental/overhead_matching/swag/farfield/catalog",
        "//experimental/overhead_matching/swag/farfield/catalog:schema",
        "//experimental/overhead_matching/swag/farfield/extraction:vertex_batch_manager_lib",
        "//experimental/overhead_matching/swag/farfield/localization:structs",
        "//experimental/overhead_matching/swag/farfield/tracking:tracklets",
    ],
)

py_binary(
    name = "match_landmarks",
    srcs = ["match_landmarks.py"],
    main = "match_landmarks.py",
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":match_landmarks_lib"],
)

py_test(
    name = "match_landmarks_test",
    srcs = ["match_landmarks_test.py"],
    deps = [
        requirement("msgspec"),
        requirement("shapely"),
        ":match_landmarks_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
        "//experimental/overhead_matching/swag/farfield/catalog:schema",
        "//experimental/overhead_matching/swag/farfield/localization:structs",
    ],
)

py_library(
    name = "match_viewer_lib",
    srcs = ["match_viewer.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pillow"),
        ":match_landmarks_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:nominal_forward",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
        "//experimental/overhead_matching/swag/farfield/calibration:heading",
        "//experimental/overhead_matching/swag/farfield/catalog",
        "//experimental/overhead_matching/swag/farfield/tracking:tracklets",
    ],
)

py_binary(
    name = "match_viewer",
    srcs = ["match_viewer.py"],
    main = "match_viewer.py",
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":match_viewer_lib"],
)

py_test(
    name = "match_viewer_test",
    srcs = [
        "match_landmarks_test.py",
        "match_viewer_test.py",
    ],
    deps = [
        requirement("msgspec"),
        requirement("pillow"),
        requirement("shapely"),
        ":match_landmarks_lib",
        ":match_viewer_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:geometry",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:nominal_forward",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:testing",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
        "//experimental/overhead_matching/swag/farfield/catalog:schema",
        "//experimental/overhead_matching/swag/farfield/localization:structs",
    ],
)
