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

cc_library(
  name = "visual_odometry",
  hdrs = ["visual_odometry.hh"],
  visibility = ["//visibility:public"],
  srcs = ["visual_odometry.cc"],
  deps = [
    "@opencv//:opencv",
    "//common:check"
  ]
)

cc_test(
  name = "visual_odometry_test",
  srcs = ["visual_odometry_test.cc"],  
  deps = [
    "@com_google_googletest//:gtest_main",
    ":visual_odometry"
  ]
)
