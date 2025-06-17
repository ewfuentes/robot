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
    "@opencv//:opencv"
  ]
)

cc_library(
  name = "symphony_lake_parser",
  hdrs = ["symphony_lake_parser.hh"],
  copts = ["-Wno-unused-parameter"],
  visibility = ["//visibility:public"],
  srcs = ["symphony_lake_parser.cc"],
  deps = [
    "@symphony_lake_parser"
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

cc_test(
  name = "symphony_lake_parser_test",
  srcs = ["symphony_lake_parser_test.cc"],
  copts = ["-Wno-unused-parameter"],
  data = ["@symphony_lake_snippet//:files"],
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
  ]
)

cc_library(
    name = "image_point", 
    hdrs = ["image_point.hh"],
    deps = [
      "//common/liegroups:se3",
      "@opencv//:opencv",
      "@eigen"
    ]
)

cc_library(
    name = "four_seasons_parser", 
    hdrs = ["four_seasons_parser.hh"],
    srcs = ["four_seasons_parser.cc"],
    deps = [
      "@eigen",
      "@opencv//:opencv",
      "//common/liegroups:se3",
      "//common/time:robot_time",
      "@com_google_absl//absl/strings:str_format",
      ":image_point"
    ]
)

cc_binary(
  name = "four_seasons_parser_example",
  srcs = ["four_seasons_parser_example.cc"],
  deps = [
    ":four_seasons_parser",
    "@cxxopts//:cxxopts",
    "//common:check"
  ]
)

cc_test(
  name = "four_seasons_parser_test",
  srcs = ["four_seasons_parser_test.cc"],
  data = ["@four_seasons_snippet//:files"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "//common/liegroups:se3",
    "@com_google_absl//absl/strings:str_format",
    "@eigen",
    ":four_seasons_parser"
  ]
)