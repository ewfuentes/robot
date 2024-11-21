
package(features=["warning_compile_flags"])

cc_library(
  name = "learn_descriptors",
  hdrs = ["learn_descriptors.hh"],
  visibility = ["//visibility:public"],
  srcs = ["learn_descriptors.cc"],
)

cc_test(
  name = "learn_descriptors_test",
  srcs = ["learn_descriptors_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":learn_descriptors",
  ]
)

cc_test(
  name = "gtsam_test",
  srcs = ["gtsam_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "@gtsam//:gtsam",
    ":learn_descriptors"  
  ]
)