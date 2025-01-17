package(features=["warning_compile_flags"])

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

cc_library(
    name = "spectacular_log",
    hdrs = ["spectacular_log.hh"],
    srcs = ["spectacular_log.cc"],
    deps = [
	"@eigen",
	"@nlohmann_json//:json",
	"@opencv",
	"@fmt",
	"//common:argument_wrapper",
	"//common:check",
	"//common/time:robot_time",
	"//common/math:cubic_hermite_spline",
    ]
)

cc_test(
    name = "spectacular_log_test",
    srcs = ["spectacular_log_test.cc"],
    data = ["@spectacular_log_snippet//:files"],
    deps = [
	"//common:matplotlib",
	":spectacular_log",
	"@com_google_googletest//:gtest_main",
	"@fmt",
    "//common/video:image_compare",
    ]
)

pybind_extension(
    name = "spectacular_log_python",
    srcs = ["spectacular_log_python.cc"],
    data = ["//common/time:robot_time_python.so"],
    deps = [":spectacular_log"],
)

py_binary(
    name = "spectacular_log_to_rosbag",
    srcs = ["spectacular_log_to_rosbag.py"],
    data = [":spectacular_log_python.so"],
    deps = [
	requirement("rosbags"),
	requirement("numpy"),
    ],
)


cc_library(
    name = "kimera_spectacular_data_provider",
    hdrs = ["kimera_spectacular_data_provider.hh"],
    srcs = ["kimera_spectacular_data_provider.cc"],
    copts = ["-Wno-sign-compare", "-Wno-vla"],
    deps = [
        "@kimera_vio",
        ":spectacular_log",
    ]
)
cc_test(
    name = "kimera_spectacular_data_provider_test",
    srcs = ["kimera_spectacular_data_provider_test.cc"],
    data = ["@spectacular_log_snippet//:files"],
    copts = ["-Wno-sign-compare", "-Wno-vla"],
    deps = [
        ":kimera_spectacular_data_provider",
        "@com_google_googletest//:gtest_main",
        "@fmt",
        "//common/video:image_compare",
    ]
)

cc_binary(
    name = "kimera_vio_pipeline_test",
    srcs = ["kimera_vio_pipeline_test.cc"],
    copts = ["-Wno-sign-compare", "-Wno-vla"],
    deps = [
        ":kimera_spectacular_data_provider",
        "@com_google_googletest//:gtest_main",
        "@fmt",
    ]
)

py_library(
    name = "grounding_sam",
    srcs = ["grounding_sam.py"],
    visibility = ["//common/python:__pkg__"],
    deps = [
	"//common:torch",
	requirement("sam-2"),
	requirement("transformers"),
	requirement("numpy"),
    ]
)

py_library(
    name = "verde",
    srcs = ["verde.py"],
    deps = [
	":grounding_sam",
	requirement("numpy"),
	requirement("matplotlib"),
	requirement("pandas"),
	requirement("supervision"),
	requirement("seaborn"),
	requirement("networkx"),
    ]
)

py_test(
    name = "verde_test",
    srcs = ["verde_test.py"],
    data = ["@cvusa_minisubset//:files"],
    deps = [
	":verde",
	requirement("Pillow"),
    ],
)

py_library(
    name = "cvusa_evaluation_pipeline",
    srcs = ["cvusa_evaluation_pipeline.py"],
    deps = [
	requirement('numpy'),
	requirement('tqdm'),
    ]
)

py_test(
    name = "cvusa_evaluation_pipeline_test",
    srcs = ["cvusa_evaluation_pipeline_test.py"],
    data = ["@cvusa_minisubset//:files"],
    deps = [
	requirement("numpy"),
	requirement("Pillow"),
	":cvusa_evaluation_pipeline",
    ]
)

py_binary(
    name = "evaluate_verde",
    srcs = ["evaluate_verde.py"],
    deps = [
	":cvusa_evaluation_pipeline",
	":verde",
    ]
)
