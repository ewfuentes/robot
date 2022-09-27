
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