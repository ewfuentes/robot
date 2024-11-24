
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
