
package(default_visibility=["//visibility:public"])

load("@//common/proto:proto.bzl", "multi_proto_library")

cc_library(
  name = "sim_clock",
  hdrs = ["sim_clock.hh"],
  srcs = ["sim_clock.cc"],
)

cc_test(
  name = "sim_clock_test",
  srcs= ["sim_clock_test.cc"],
  deps = [
    ":sim_clock",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "robot_time",
  hdrs = ["robot_time.hh"],
  srcs = ["robot_time.cc"],
  deps = [
   ":sim_clock",
  ]
)

cc_test(
  name = "robot_time_test",
  srcs = ["robot_time_test.cc"],
  deps = [
    ":robot_time",
    "@com_google_googletest//:gtest_main",
   ":sim_clock",
  ]
)

multi_proto_library(
  name = "robot_time_proto",
  srcs= ["robot_time.proto"],
)

cc_library(
  name = "robot_time_to_proto",
  hdrs = ["robot_time_to_proto.hh"],
  srcs = ["robot_time_to_proto.cc"],
  deps = [
  ":robot_time_proto",
  ":robot_time",
  ]
)

cc_test(
  name = "robot_time_to_proto_test",
  srcs = ["robot_time_to_proto_test.cc"],
  deps = [
    ":robot_time_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)
