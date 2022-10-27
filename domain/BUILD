
cc_library(
  name = "tiger",
  hdrs = ["tiger.hh"],
  srcs = ["tiger.cc"],
)

cc_test(
  name = "tiger_test",
  srcs = ["tiger_test.cc"],
  deps = [
    ":tiger",
    "@com_google_googletest//:gtest_main",
  ]
)