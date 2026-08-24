load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "llm_cost",
    srcs = ["llm_cost.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
)

py_test(
    name = "llm_cost_test",
    srcs = ["llm_cost_test.py"],
    deps = [":llm_cost"],
)

py_library(
    name = "vertex_batch_manager_lib",
    srcs = ["vertex_batch_manager.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("google-cloud-storage"),
        requirement("google-genai"),
        ":llm_cost",
        ":prompts",
    ],
)

py_binary(
    name = "vertex_batch_manager",
    srcs = ["vertex_batch_manager.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":vertex_batch_manager_lib"],
)

py_test(
    name = "vertex_batch_manager_test",
    srcs = ["vertex_batch_manager_test.py"],
    deps = [
        ":llm_cost",
        ":prompts",
        ":vertex_batch_manager_lib",
    ],
)

py_library(
    name = "panorama_to_pinhole_lib",
    srcs = ["panorama_to_pinhole.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        requirement("scipy"),
        requirement("tqdm"),
    ],
)

py_binary(
    name = "panorama_to_pinhole",
    srcs = ["panorama_to_pinhole.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":panorama_to_pinhole_lib"],
)

py_library(
    name = "prompts",
    srcs = ["prompts.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pydantic"),
        requirement("tqdm"),
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)

py_test(
    name = "prompts_test",
    srcs = ["prompts_test.py"],
    deps = [":prompts"],
)

py_library(
    name = "extract_landmarks_lib",
    srcs = ["extract_landmarks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("numpy"),
        requirement("pillow"),
        ":llm_cost",
        ":panorama_to_pinhole_lib",
        ":prompts",
        ":vertex_batch_manager_lib",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:dataset",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:publication",
    ],
)

py_binary(
    name = "extract_landmarks",
    srcs = ["extract_landmarks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":extract_landmarks_lib"],
)

py_test(
    name = "extract_landmarks_test",
    srcs = ["extract_landmarks_test.py"],
    deps = [
        requirement("pillow"),
        ":extract_landmarks_lib",
        ":prompts",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:build_config",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:testing",
        "//experimental/overhead_matching/swag/farfield/collection:run_farfield_collection_lib",
    ],
)

py_library(
    name = "legacy_extraction_adoption_lib",
    srcs = ["legacy_extraction_adoption.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pillow"),
        ":extract_landmarks_lib",
        ":prompts",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)

py_binary(
    name = "legacy_extraction_adoption",
    srcs = ["legacy_extraction_adoption.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [":legacy_extraction_adoption_lib"],
)

py_test(
    name = "legacy_extraction_adoption_test",
    srcs = ["legacy_extraction_adoption_test.py"],
    deps = [
        requirement("pillow"),
        ":legacy_extraction_adoption_lib",
        ":prompts",
        "//experimental/overhead_matching/swag/farfield:artifact",
        "//experimental/overhead_matching/swag/farfield:llm_lifecycle",
        "//experimental/overhead_matching/swag/farfield:paths",
    ],
)
