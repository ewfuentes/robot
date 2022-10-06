
load("//common/proto:proto.bzl", "multi_proto_library")

multi_proto_library(
  name = "sim_log_proto",
  srcs = ["sim_log.proto"],
  visibility = [":__subpackages__"],
  deps = [
    ":beacon_sim_debug_proto",
  ]
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
    ":beacon_observation_to_proto",
    ":beacon_sim_debug_proto",
    ":ekf_slam_estimate_to_proto",
    "//common/time:sim_clock",
    ":ekf_slam",
    "//visualization/gl_window",
    "//common:argument_wrapper",
    "//common/liegroups:se2",
    "//common/liegroups:se2_to_proto",
    "//common/liegroups:se3",
    "//common/time:robot_time_to_proto",
    "@cxxopts//:cxxopts",
  ],
)

multi_proto_library(
  name = "beacon_sim_debug_proto",
  srcs = ["beacon_sim_debug.proto"],
  deps = [
    ":ekf_slam_estimate_proto",
    ":beacon_observation_proto",
    "//common/time:robot_time_proto",
    "//common/liegroups:se2_proto",
  ]
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

multi_proto_library(
  name = "ekf_slam_estimate_proto",
  srcs = ["ekf_slam_estimate.proto"],
  deps = [
    "//common/math:matrix_proto",
  ]
)

cc_library(
  name = "ekf_slam_estimate_to_proto",
  hdrs = ["ekf_slam_estimate_to_proto.hh"],
  srcs = ["ekf_slam_estimate_to_proto.cc"],
  deps = [
    ":ekf_slam",
    ":ekf_slam_estimate_proto",
    "//common/math:matrix_to_proto",
  ]
)

cc_test(
  name = "ekf_slam_estimate_to_proto_test",
  srcs = ["ekf_slam_estimate_to_proto_test.cc"],
  deps = [
    ":ekf_slam_estimate_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)

multi_proto_library(
  name = "beacon_observation_proto",
  srcs = ["beacon_observation.proto"],
)

cc_library(
  name = "beacon_observation_to_proto",
  hdrs = ["beacon_observation_to_proto.hh"],
  srcs = ["beacon_observation_to_proto.cc"],
  deps = [
    ":beacon_observation_proto",
    ":generate_observations",
  ]
)

cc_test(
  name = "beacon_observation_to_proto_test",
  srcs = ["beacon_observation_to_proto_test.cc"],
  deps = [
    ":beacon_observation_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)
