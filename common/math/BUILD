
load("@//common/proto:proto.bzl", "multi_proto_library")


multi_proto_library(
  name = "matrix_proto",
  srcs = ["matrix.proto"],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "matrix_to_proto",
  hdrs = ["matrix_to_proto.hh"],
  deps = [
    "@eigen//:eigen",
    ":matrix_proto",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "matrix_to_proto_test",
  srcs = ["matrix_to_proto_test.cc"],
  deps = [
    ":matrix_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "redheffer_star",
  hdrs = ["redheffer_star.hh"],
  srcs = ["redheffer_star.cc"],
  visibility = ["//visibility:public"],
  deps = [
    "@eigen//:eigen",
  ]
)

cc_test(
  name = "redheffer_star_test",
  srcs = ["redheffer_star_test.cc"],
  deps = [
    ":redheffer_star",
    "@com_google_googletest//:gtest_main",
  ]
)
