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
