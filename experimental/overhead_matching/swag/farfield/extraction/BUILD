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
        "//experimental/overhead_matching/swag/model:farfield_prompt_text",
        requirement("pydantic"),
        requirement("tqdm"),
    ],
)

py_test(
    name = "prompts_test",
    srcs = ["prompts_test.py"],
    data = ["//experimental/overhead_matching/swag/model:semantic_landmark_extractor_src"],
    deps = [":prompts"],
)

py_library(
    name = "extract_landmarks_lib",
    srcs = ["extract_landmarks.py"],
    visibility = ["//experimental/overhead_matching/swag:__subpackages__"],
    deps = [
        requirement("pillow"),
        ":llm_cost",
        ":panorama_to_pinhole_lib",
        ":prompts",
        ":vertex_batch_manager_lib",
        "//experimental/overhead_matching/swag/farfield:paths",
        "//experimental/overhead_matching/swag/farfield:provenance",
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
        "//experimental/overhead_matching/swag/farfield:provenance",
        "//experimental/overhead_matching/swag/farfield:testing",
    ],
)
