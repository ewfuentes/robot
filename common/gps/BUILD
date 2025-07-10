package(features=["warning_compile_flags"])

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

cc_library(
  name = "frame_translation",
  srcs = ["frame_translation.cc"],
  hdrs = ["frame_translation.hh"],
  visibility = ["//visibility:public"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "@geographiclib//:geographiclib",
    "@eigen"
  ]
)

cc_test(
  name = "frame_translation_test",
  srcs = ["frame_translation_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":frame_translation",
    "@eigen",
    "//common:check"
  ]
)