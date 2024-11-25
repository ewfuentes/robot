

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
	":spectacular_log",
	"@com_google_googletest//:gtest_main",
	"@fmt",
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
	requirement("ipdb"),
    ],
)
