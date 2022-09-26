
load("//common/proto:proto.bzl", "multi_proto_library")

multi_proto_library(
  name = "sim_log_proto",
  srcs = ["sim_log.proto"],
)

cc_binary(
  name = "beacon_sim",
  srcs = ["beacon_sim.cc"],
  linkopts = ["-latomic", "-lGLU"],
  deps = [
    ":world_map",
    ":robot",
    ":sim_log_proto",
    ":generate_observations",
    "//common/time:sim_clock",
    ":ekf_slam",
    "//visualization/gl_window",
    "//common:argument_wrapper",
    "//common/liegroups:se2",
    "//common/liegroups:se3",
    "@cxxopts//:cxxopts",
  ],
)

cc_library(
  name = "world_map",
  hdrs = ["world_map.hh"],
  srcs = ["world_map.cc"],
  deps = [
    "//common/time:robot_time",
    "@eigen//:eigen",
  ],
)

cc_test(
  name = "world_map_test",
  srcs = ["world_map_test.cc"],
  deps = [
    ":world_map",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "robot",
  hdrs = ["robot.hh"],
  srcs = ["robot.cc"],
  deps = [
    "@eigen//:eigen",
    "//common/liegroups:se2",
  ]
)

cc_test(
  name = "robot_test",
  srcs = ["robot_test.cc"],
  deps = [":robot",
  "@com_google_googletest//:gtest_main"],
)

cc_library(
  name = "generate_observations",
  hdrs = ["generate_observations.hh"],
  srcs = ["generate_observations.cc"],
  deps = [
    ":world_map",
    ":robot",
    "//common:argument_wrapper",
  ],
)

cc_test(
  name = "generate_observations_test",
  srcs = ["generate_observations_test.cc"],
  deps = [
    ":generate_observations",
    "@com_google_googletest//:gtest_main"
  ]
)

cc_library(
  name = "ekf_slam",
  hdrs = ["ekf_slam.hh"],
  srcs = ["ekf_slam.cc"],
  deps = [
    ":generate_observations",
    "@eigen//:eigen",
    "//common/liegroups:se2",
  ]
)

cc_test(
  name = "ekf_slam_test",
  srcs = ["ekf_slam_test.cc"],
  deps = [
    ":ekf_slam",
    "@com_google_googletest//:gtest_main",
  ]
)

multi_proto_library(
  name = "world_map_config_proto",
  srcs = ["world_map_config.proto"],
)

cc_library(
  name = "world_map_config_to_proto",
  hdrs = ["world_map_config_to_proto.hh"],
  srcs = ["world_map_config_to_proto.cc"],
  deps = [
    ":world_map_config_proto",
    ":world_map",
  ]
)

cc_test(
  name = "world_map_config_to_proto_test",
  srcs = ["world_map_config_to_proto_test.cc"],
  deps = [
    ":world_map_config_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)
