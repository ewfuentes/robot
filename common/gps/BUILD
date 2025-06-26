package(features=["warning_compile_flags"])
load("@pip//:requirements.bzl", "requirement")

cc_test(
  name = "nmea_test",
  srcs = ["nmea_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "@nmea",
  ]
)

cc_test(
  name = "geographiclib_test",
  srcs = ["geographiclib_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "@geographiclib//:geographiclib",
  ]
)

py_library(
  name = "web_mercator",
  srcs = ["web_mercator.py"],
  visibility = ["//visibility:public"],
  deps = [
    requirement("numpy"),
    requirement("torch"),
    "//common/torch:load_torch_deps",
  ]
)

py_test(
  name = "web_mercator_test",
  srcs = ["web_mercator_test.py"],
  deps = [":web_mercator"],
)
