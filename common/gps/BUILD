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