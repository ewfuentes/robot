
load("//common/proto/proto.bzl", "multi_proto_library")

package(default_visibility=["//visibility:public"])

cc_library(
  name = "so2",
  hdrs = ["so2.hh"],
  deps = [
    "@sophus//:liegroups",
  ]
)


cc_library(
  name = "se2",
  hdrs = ["se2.hh"],
  srcs = ["se2.cc"],
  deps = [
    ":so2",
    "@sophus//:liegroups",
  ]
)

cc_test(
  name = "se2_test",
  srcs = ["se2_test.cc"],
  deps = [
    ":se2",
    "@com_google_googletest//:gtest_main",
  ],
)

cc_library(
  name = "se3",
  hdrs = ["se3.hh"],
  srcs = ["se3.cc"],
  deps = [
    "@sophus//:liegroups",
  ]
)

cc_test(
  name = "se3_test",
  srcs = ["se3_test.cc"],
  deps = [
    ":se3",
    "@com_google_googletest//:gtest_main",
  ],
)

multi_proto_library(
  name = "so2_proto",
  srcs = ["so2.proto"],
)

multi_proto_library(
  name = "so3_proto",
  srcs = ["so3.proto"],
)

multi_proto_library(
  name = "se2_proto",
  srcs = ["se2.proto"],
  deps = [
    ":so2_proto",
    "//common/math:matrix_proto",
  ]
)

multi_proto_library(
  name = "se3_proto",
  srcs = ["se3.proto"],
  deps = [
    ":so3_proto",
    "//common/math:matrix_proto",
  ]
)