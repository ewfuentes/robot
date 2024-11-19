
cc_library(
    name = "spectacular_data_provider",
    hdrs = ["spectacular_data_provider.hh"],
    srcs = ["spectacular_data_provider.cc"],
    deps = [
	"@kimera_vio",
	"@nlohmann_json//:json",
	"//common:check",
	"//common/time:robot_time",
    ]
)

cc_test(
    name = "spectacular_data_provider_test",
    srcs = ["spectacular_data_provider_test.cc"],
    deps = [
	":spectacular_data_provider",
	"@com_google_googletest//:gtest_main",
    ]
)
