load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "match_landmarks_lib",
    srcs = ["match_landmarks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("msgspec"),
        "//common/python:serialization",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:artifact_recipe",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:configured_lane",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
        "//experimental/overhead_matching/swag/farfield/catalog",
        "//experimental/overhead_matching/swag/farfield/catalog:lineage",
        "//experimental/overhead_matching/swag/farfield/catalog:schema",
        "//experimental/overhead_matching/swag/farfield/extraction:vertex_batch_manager_lib",
        "//experimental/overhead_matching/swag/farfield/localization:structs",
        "//experimental/overhead_matching/swag/farfield/tracking:tracklets",
        "//experimental/overhead_matching/swag/farfield/viewers:page",
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
        "//experimental/overhead_matching/swag/farfield:configured_lane",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield/calibration:audit_io",
        "//experimental/overhead_matching/swag/farfield/catalog:schema",
        "//experimental/overhead_matching/swag/farfield/localization:structs",
    ],
)

py_library(
    name = "identity_review_lib",
    srcs = ["identity_review.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:configured_lane",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
    ],
)

py_binary(
    name = "identity_review",
    srcs = ["identity_review.py"],
    main = "identity_review.py",
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":identity_review_lib"],
)

py_test(
    name = "identity_review_test",
    srcs = ["identity_review_test.py"],
    deps = [
        ":identity_review_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:configured_lane",
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)

py_library(
    name = "match_viewer_lib",
    srcs = ["match_viewer.py"],
    data = [
        "match_viewer_assets/app.js",
        "match_viewer_assets/notes.js",
        "match_viewer_assets/style.css",
    ],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pillow"),
        ":identity_review_lib",
        ":match_landmarks_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:configured_lane",
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
        "//experimental/overhead_matching/swag/farfield/viewers:page",
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
        "//experimental/overhead_matching/swag/farfield:configured_lane",
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
