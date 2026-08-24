load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "artifact",
    srcs = ["artifact.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_test(
    name = "artifact_test",
    srcs = ["artifact_test.py"],
    deps = [":artifact"],
)

py_library(
    name = "publication",
    srcs = ["publication.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":artifact",
        "//experimental/overhead_matching/swag/farfield/viewers:indexes",
    ],
)

py_test(
    name = "publication_test",
    srcs = ["publication_test.py"],
    deps = [
        ":artifact",
        ":publication",
    ],
)

py_library(
    name = "llm_lifecycle",
    srcs = ["llm_lifecycle.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":artifact"],
)

py_test(
    name = "llm_lifecycle_test",
    srcs = ["llm_lifecycle_test.py"],
    deps = [
        ":artifact",
        ":llm_lifecycle",
    ],
)

py_library(
    name = "geometry",
    srcs = ["geometry.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
    ],
)

py_test(
    name = "geometry_test",
    srcs = ["geometry_test.py"],
    deps = [
        requirement("numpy"),
        ":geometry",
    ],
)

py_library(
    name = "paths",
    srcs = ["paths.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":artifact"],
)

py_test(
    name = "paths_test",
    srcs = ["paths_test.py"],
    deps = [":paths"],
)

py_library(
    name = "provenance",
    srcs = ["provenance.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_test(
    name = "provenance_test",
    srcs = ["provenance_test.py"],
    deps = [":provenance"],
)

py_library(
    name = "build_config",
    srcs = ["build_config.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":artifact",
        ":provenance",
    ],
)

py_test(
    name = "build_config_test",
    srcs = ["build_config_test.py"],
    deps = [":build_config"],
)

py_library(
    name = "nominal_forward",
    srcs = ["nominal_forward.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":geometry"],
)

py_test(
    name = "nominal_forward_test",
    srcs = ["nominal_forward_test.py"],
    deps = [
        ":geometry",
        ":nominal_forward",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        ":artifact",
        ":geometry",
        ":paths",
    ],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    deps = [
        requirement("pillow"),
        ":dataset",
        ":geometry",
        ":testing",
    ],
)

py_library(
    name = "testing",
    testonly = True,
    srcs = ["testing.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pillow"),
        ":artifact",
        ":geometry",
        ":paths",
    ],
)

py_library(
    name = "audit_dataset_lib",
    srcs = ["audit_dataset.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python"),
        requirement("pillow"),
        ":geometry",
    ],
)

py_binary(
    name = "audit_dataset",
    srcs = ["audit_dataset.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("opencv-python"),
        requirement("pillow"),
        ":geometry",
    ],
)

py_test(
    name = "audit_dataset_test",
    srcs = ["audit_dataset_test.py"],
    deps = [
        requirement("pillow"),
        ":audit_dataset_lib",
        ":testing",
    ],
)

py_library(
    name = "pipeline_lib",
    srcs = ["pipeline.py"],
    data = ["configs/harbor_example.yaml"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pyyaml"),
        ":artifact",
        ":build_config",
        ":nominal_forward",
        ":paths",
        ":provenance",
        "//experimental/overhead_matching/swag/farfield/localization:run_identity",
        "//experimental/overhead_matching/swag/farfield/matching:identity_review_lib",
    ],
)

py_binary(
    name = "pipeline",
    srcs = ["pipeline.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":pipeline_lib"],
)

py_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.py"],
    data = ["configs/harbor_example.yaml"],
    deps = [
        requirement("pyyaml"),
        ":artifact",
        ":build_config",
        ":paths",
        ":pipeline_lib",
        "//experimental/overhead_matching/swag/farfield/matching:identity_review_lib",
    ],
)

py_test(
    name = "pipeline_cli_test",
    srcs = ["pipeline_cli_test.py"],
    data = [
        "configs/harbor_example.yaml",
        ":pipeline",
    ],
    deps = [
        requirement("pyyaml"),
        ":artifact",
        "@rules_python//python/runfiles",
    ],
)
