load("@//common/proto:proto.bzl", "multi_proto_library")

multi_proto_library(
  name = "sample_message_proto",
  srcs = ["sample_message.proto"],
)

cc_library(
  name = "load_from_file",
  hdrs = ["load_from_file.hh"],
)

cc_test(
  name = "load_from_file_test",
  srcs = ["load_from_file_test.cc"],
  deps = [
    ":sample_message_proto",
    ":load_from_file",
    "@com_google_googletest//:gtest_main",
  ]
)
