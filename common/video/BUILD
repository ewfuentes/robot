package(features=["warning_compile_flags"])

cc_library(
  name = "image_compare",
  hdrs = ["image_compare.hh"],
  srcs = ["image_compare.cc"],
  deps = [
    "@eigen//:eigen",
    "@opencv",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "image_compare_test",
  srcs = ["image_compare_test.cc"],
  deps = [
    ":image_compare",
    "@com_google_googletest//:gtest_main",
  ]
)
