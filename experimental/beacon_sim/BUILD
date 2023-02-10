load("@pip//:requirements.bzl", "requirement")
load("//common/proto:proto.bzl", "multi_proto_library")

multi_proto_library(
  name = "sim_log_proto",
  srcs = ["sim_log.proto"],
  deps = [
    ":beacon_sim_debug_proto",
  ]
)

cc_library(
  name = "tick_sim",
  hdrs = ["tick_sim.hh"],
  srcs = ["tick_sim.cc"],
  deps = [
    "//common:argument_wrapper",
    "//common/time:robot_time_to_proto",
    "//common/liegroups:se2_to_proto",
    ":beacon_sim_state",
    ":sim_config",
    ":beacon_sim_debug_proto",
    ":ekf_slam_estimate_to_proto",
    ":beacon_observation_to_proto",
  ]
)

cc_library(
  name = "sim_config",
  hdrs = ["sim_config.hh"],
)

cc_binary(
  name = "beacon_sim",
  srcs = ["beacon_sim.cc"],
  linkopts = ["-latomic"],
  deps = [
    ":sim_config",
    ":world_map",
    ":robot",
    ":sim_log_proto",
    ":generate_observations",
    ":beacon_observation_to_proto",
    ":beacon_sim_debug_proto",
    ":beacon_sim_state",
    ":belief_road_map_planner",
    ":ekf_slam_estimate_to_proto",
    ":extract_mapped_landmarks",
    ":mapped_landmarks_to_proto",
    "//common/time:sim_clock",
    ":ekf_slam",
    ":tick_sim",
    ":visualize_beacon_sim",
    "//visualization/gl_window",
    "//common:argument_wrapper",
    "//common/liegroups:se2",
    "//common/liegroups:se2_to_proto",
    "//common/proto:load_from_file",
    "//common/time:robot_time_to_proto",
    "//planning:probabilistic_road_map",
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
    ":obstacle",
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
    ":mapped_landmarks",
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
    "//common/time:robot_time_proto",
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
    "//common/time:robot_time_to_proto",
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

cc_library(
  name = "obstacle",
  hdrs = ["obstacle.hh"],
  srcs = ["obstacle.cc"],
  deps = [
    "@eigen//:eigen",
    "//common/liegroups:se2",
  ]
)

cc_test(
  name = "obstacle_test",
  srcs = ["obstacle_test.cc"],
  deps = [
    ":obstacle",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "mapped_landmarks",
  hdrs = ["mapped_landmarks.hh"],
  deps = [
    ":world_map",
    "@eigen//:eigen",
  ]
)

cc_library(
  name = "extract_mapped_landmarks",
  hdrs = ["extract_mapped_landmarks.hh"],
  srcs = ["extract_mapped_landmarks.cc"],
  deps = [
    ":mapped_landmarks",
    ":ekf_slam",
  ],
)

cc_test(
  name = "extract_mapped_landmarks_test",
  srcs = ["extract_mapped_landmarks_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":extract_mapped_landmarks",
  ]
)

multi_proto_library(
  name = "mapped_landmarks_proto",
  srcs = ["mapped_landmarks.proto"],
  deps = [
    ":world_map_config_proto",
    "//common/math:matrix_proto",
  ]
)

cc_library(
  name = "mapped_landmarks_to_proto",
  hdrs = ["mapped_landmarks_to_proto.hh"],
  srcs = ["mapped_landmarks_to_proto.cc"],
  deps = [
    ":world_map_config_to_proto",
    "//common/math:matrix_to_proto",
    ":mapped_landmarks_proto",
    ":mapped_landmarks",
  ],
)

cc_test(
  name = "mapped_landmarks_to_proto_test",
  srcs = ["mapped_landmarks_to_proto_test.cc"],
  deps = [
    ":mapped_landmarks_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "visualize_beacon_sim",
  hdrs = ["visualize_beacon_sim.hh"],
  srcs = ["visualize_beacon_sim.cc"],
  linkopts = ["-lGLU"],
  deps = [
    ":beacon_sim_state",
    "//common/liegroups:se3",
  ]
)

cc_library(
  name = "beacon_sim_state",
  hdrs = ["beacon_sim_state.hh"],
  deps = [
    ":world_map",
    ":ekf_slam",
    ":robot",
    ":generate_observations",
    ":belief_road_map_planner",
    "//common/time:robot_time",
    "//planning:probabilistic_road_map",
  ]
)

cc_library(
  name = "belief_road_map_planner",
  hdrs = ["belief_road_map_planner.hh"],
  srcs = ["belief_road_map_planner.cc"],
  deps = [
    ":ekf_slam",
    ":robot",
    "//planning:belief_road_map",
    "//planning:probabilistic_road_map",
    "//common/math:redheffer_star",
    ":generate_observations",
  ]
)

cc_test(
  name = "belief_road_map_planner_test",
  srcs = ["belief_road_map_planner_test.cc"],
  deps = [
    ":belief_road_map_planner",
    "@com_google_googletest//:gtest_main",
  ],
)

cc_binary(
  name = "run_trials",
  srcs = ["run_trials.cc"],
  linkopts = ["-ltbb"],
  deps = [
    ":beacon_sim_state",
    ":ekf_slam",
    ":robot",
    ":world_map",
    ":mapped_landmarks",
    ":sim_config",
    ":belief_road_map_planner",
    ":tick_sim",
    ":rollout_statistics_proto",
    ":world_map_config_to_proto",
    "//planning:probabilistic_road_map",
    "//planning:road_map_to_proto",
    "//common/liegroups:se2",
    "//common/liegroups:se2_to_proto",
    "//common/time:robot_time",
    "@cxxopts//:cxxopts",
  ],
)

py_binary(
  name = "plot_trials",
  srcs = ["plot_trials.py"],
  visibility = ["//visibility:public"],
  deps = [
    ":rollout_statistics_proto_py",
    requirement("matplotlib"),
    requirement("numpy"),
  ]
)

multi_proto_library(
  name = "rollout_statistics_proto",
  srcs = ["rollout_statistics.proto"],
  deps = [
    ":beacon_sim_debug_proto",
    ":world_map_config_proto",
    "//planning:road_map_proto",
    "//common/liegroups:se2_proto",
  ]
)