package(features=["warning_compile_flags"])

cc_library(
  name = "csv_reader",
  hdrs = ["learn_descriptorcsv.hh"],
  visibility = ["//visibility:public"],
  srcs = ["learn_descriptors.cc"],
  deps = [
    "@"
  ]
)

cc_test(
  name = "csv_reader_test",
  srcs = ["csv_reader_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":learn_descriptors",
  ]
)
