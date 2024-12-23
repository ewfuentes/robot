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
  name = "structure_from_motion",
  hdrs = ["structure_from_motion.hh"],
  visibility = ["//visibility:public"],
  srcs = ["structure_from_motion.cc"],
  deps = [
    "@opencv//:opencv",
    "@gtsam//:gtsam",
    "@eigen"
  ]
)

cc_library(
  name = "symphony_lake_parser",
  hdrs = ["symphony_lake_parser.hh"],
  copts = [
    "-Wno-unused-parameter",
    "-std=c++20",
  ],
  data = ["@symphony_lake_snippet"],
  visibility = ["//visibility:public"],
  srcs = ["symphony_lake_parser.cc"],
  deps = [
    "@symphony_lake_parser",
    "//common:check",
    "@eigen"
  ]
)

cc_test(
  name = "structure_from_motion_test",
  srcs = ["structure_from_motion_test.cc"],  
  copts = [
    "-Wno-unused-parameter",
  ],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":symphony_lake_parser",
    ":structure_from_motion"
  ]
)

cc_test(
  name = "symphony_lake_parser_test",
  srcs = ["symphony_lake_parser_test.cc"],
  copts = ["-Wno-unused-parameter"],
  # data = ["@symphony_lake_snippet"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":symphony_lake_parser",
    "//common:check"
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
