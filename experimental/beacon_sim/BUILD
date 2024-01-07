load("@pip//:requirements.bzl", "requirement")
load("//common/proto:proto.bzl", "multi_proto_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

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
    ":belief_road_map_planner",
    ":beacon_sim_state",
    ":information_lower_bound_search",
    ":sim_config",
    ":beacon_sim_debug_proto",
    ":ekf_slam_estimate_to_proto",
    ":beacon_observation_to_proto",
    ":information_lower_bound_planner",
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
    ":world_map_config_to_proto",
    ":robot",
    ":sim_log_proto",
    ":generate_observations",
    ":beacon_observation_to_proto",
    ":beacon_sim_debug_proto",
    ":beacon_sim_state",
    ":belief_road_map_planner",
    ":correlated_beacons",
    ":ekf_slam_estimate_to_proto",
    ":extract_mapped_landmarks",
    ":mapped_landmarks_to_proto",
    "//common/time:sim_clock",
    ":ekf_slam",
    ":tick_sim",
    ":visualize_beacon_sim",
    "//visualization/gl_window",
    "//common:argument_wrapper",
    "//common:check",
    "//common/liegroups:se2",
    "//common/liegroups:se2_to_proto",
    "//common/proto:load_from_file",
    "//common/time:robot_time_to_proto",
    "//planning:probabilistic_road_map",
    "//planning:road_map_to_proto",
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
    "//common:argument_wrapper",
    "//common/time:robot_time",
    "@eigen//:eigen",
    ":correlated_beacons",
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
    "//common:check",
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
  deps = [
    ":beacon_potential_proto",
  ]
)

cc_library(
  name = "world_map_config_to_proto",
  hdrs = ["world_map_config_to_proto.hh"],
  srcs = ["world_map_config_to_proto.cc"],
  deps = [
    ":world_map_config_proto",
    ":world_map",
    # ":correlated_beacons_to_proto",
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
    ":correlated_beacons",
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
    ":generate_observations",
    ":robot",
    ":robot_belief",
    ":make_belief_updater",
    "//common:check",
    "//common/math:combinations",
    "//planning:belief_road_map",
    "//planning:probabilistic_road_map",
  ]
)

cc_test(
  name = "belief_road_map_planner_test",
  srcs = ["belief_road_map_planner_test.cc"],
  deps = [
    ":belief_road_map_planner",
    ":test_helpers",
    "//common:check",
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

py_binary(
  name = "run_many_trials",
  srcs = ["run_many_trials.py"],
  data = [":run_trials"],
  deps = [
    requirement("matplotlib")
  ]
)

cc_library(
  name = "log_marginal",
  hdrs = ["log_marginal.hh"],
)

cc_library(
  name = "beacon_potential",
  hdrs = [
    "beacon_potential.hh",
    "beacon_potential_to_proto.hh",
  ],
  srcs = [
    "beacon_potential.cc",
    "beacon_potential_to_proto.cc",
  ],
  deps = [
    "@eigen",
    "//common:argument_wrapper",
    "//common:check",
    ":log_marginal",
    ":beacon_potential_proto",
    ":precision_matrix_potential",
    ":correlated_beacon_potential",
  ],
)

cc_library(
  name = "precision_matrix_potential",
  hdrs = [
    "precision_matrix_potential.hh",
    "precision_matrix_potential_to_proto.hh",
  ],
  srcs = [
    "precision_matrix_potential.cc",
    "precision_matrix_potential_to_proto.cc",
  ],
  deps = [
    "@eigen",
    ":log_marginal",
    ":precision_matrix_potential_proto",
    ":beacon_potential_proto",
    "//common:check",
    "//common/math:combinations",
    "//common:argument_wrapper",
    "//common/math:matrix_to_proto",
  ]
)


cc_test(
    name = "beacon_potential_to_proto_test",
    srcs = ["beacon_potential_to_proto_test.cc"],
    deps = [
      "@com_google_googletest//:gtest_main",
      ":beacon_potential",
      ":correlated_beacons",
    ]
)

multi_proto_library(
  name = "correlated_beacon_potential_proto",
  srcs = ["correlated_beacon_potential.proto"],
)

cc_library(
  name = "correlated_beacon_potential",
  hdrs = [
    "correlated_beacon_potential.hh",
    "correlated_beacon_potential_to_proto.hh",
  ],
  srcs = [
    "correlated_beacon_potential.cc",
    "correlated_beacon_potential_to_proto.cc",
  ],
  deps = [
    ":beacon_potential_proto",
    ":correlated_beacon_potential_proto",
    ":log_marginal",
    "//common:argument_wrapper",
    "//common:check",
    "//common/math:combinations",
    "//common/math:n_choose_k",
  ]
)

cc_test(
  name = "correlated_beacon_potential_test",
  srcs = ["correlated_beacon_potential_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":correlated_beacon_potential",
  ]
)

cc_test(
  name = "correlated_beacon_potential_to_proto_test",
  srcs = ["correlated_beacon_potential_to_proto_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":correlated_beacon_potential",
  ]
)

cc_library(
  name = "correlated_beacons",
  hdrs = ["correlated_beacons.hh"],
  srcs = ["correlated_beacons.cc"],
  deps = [
    ":beacon_potential",
    "@eigen//:eigen",
    "//common:check",
    "//common:drake",
    "//common/math:combinations",
    "//common/math:n_choose_k",
  ]
)

pybind_extension(
  name = "correlated_beacons_python",
  srcs = ["correlated_beacons_python.cc"],
  visibility=["//common/python:__subpackages__"],
  deps = [
    ":correlated_beacons",
    ":beacon_potential",
    ":correlated_beacon_potential",
  ]
)

py_test(
  name = "correlated_beacons_test",
  srcs = ["correlated_beacons_test.py"],
  data = [
    ":correlated_beacons_python.so",
  ],
  deps = [
    requirement("numpy"),
  ]
)

cc_test(
  name = "drake_optimization_test",
  srcs = ["drake_optimization_test.cc"],
  deps = [
    "//common:drake",
    "@com_google_googletest//:gtest_main",
  ]
)

multi_proto_library(
  name = "beacon_potential_proto",
  srcs = ["beacon_potential.proto"],
  deps = [
    ":precision_matrix_potential_proto",
    ":correlated_beacon_potential_proto",
  ]
)

multi_proto_library(
  name = "precision_matrix_potential_proto",
  srcs = ["precision_matrix_potential.proto"],
  deps = [
    "//common/math:matrix_proto",
  ]
)

cc_test(
  name = "precision_matrix_potential_to_proto_test",
  srcs = ["precision_matrix_potential_to_proto_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":beacon_potential",
  ]
)

cc_library(
  name = "make_belief_updater",
  srcs = ["make_belief_updater.cc"],
  hdrs = ["make_belief_updater.hh"],
  deps = [
    "@eigen",
    ":ekf_slam",
    ":correlated_beacons",
    ":robot_belief",
    "//common:check",
    "//common/liegroups:se2",
    "//common/geometry:nearest_point_on_segment",
    "//common/math:combinations",
    "//common/math:redheffer_star",
    "//common/math:sample_without_replacement",
    "//planning:belief_road_map",
    "//planning:probabilistic_road_map",
  ]
)

cc_test(
  name = "make_belief_updater_test",
  srcs = ["make_belief_updater_test.cc"],
  deps = [
    ":make_belief_updater",
    ":test_helpers",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "make_eigen_bounds",
  srcs = ["make_eigen_bounds.cc"],
  hdrs = ["make_eigen_bounds.hh"],
  deps = [
    "@eigen",
    ":make_belief_updater"
  ]
)

cc_test(
  name = "make_eigen_bounds_test",
  srcs = ["make_eigen_bounds_test.cc"],
  deps = [
    ":make_eigen_bounds",
    ":test_helpers",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "robot_belief",
  hdrs = ["robot_belief.hh"],
  deps = [
    "//common/liegroups:se2",
    "@eigen",
  ]
)

cc_library(
    name = "test_helpers",
    hdrs = ["test_helpers.hh"],
    srcs = ["test_helpers.cc"],
    deps = [
      ":ekf_slam",
      ":beacon_potential",
      "//planning:probabilistic_road_map",
    ]
)

cc_library(
    name = "information_lower_bound_search",
    hdrs = ["information_lower_bound_search.hh"],
    srcs = ["information_lower_bound_search.cc"],
    deps = [
      "@eigen",
      "//common:check",
      "//planning:probabilistic_road_map",
    ]
)

cc_test(
    name = "information_lower_bound_search_test",
    srcs = ["information_lower_bound_search_test.cc"],
    deps = [
      ":information_lower_bound_search",
      "@com_google_googletest//:gtest_main",
    ]
)

cc_library(
  name = "information_lower_bound_planner",
  hdrs = ["information_lower_bound_planner.hh"],
  srcs = ["information_lower_bound_planner.cc"],
  deps = [
    ":ekf_slam",
    ":information_lower_bound_search",
    ":make_eigen_bounds",
    ":robot_belief",
    "//planning:belief_road_map",
  ]
)

py_library(
  name = "environment_builder",
  srcs = ["environment_builder.py"],
  data = ["//planning:probabilistic_road_map_python.so"],
  deps = [
    ":world_map_config_proto_py",
    ":mapped_landmarks_proto_py",
    requirement("matplotlib"),
    requirement("numpy"),
    requirement("spatialmath_python"),
  ],
  visibility =  ["//visibility:public"]
)

filegroup(
  name = "experiment_configs",
  srcs = glob(["experiment_configs/**/*.pbtxt"])
)

multi_proto_library(
  name = "experiment_config_proto",
  srcs = ["experiment_config.proto"],
)

cc_binary(
  name = "run_experiment",
  srcs = ["run_experiment.cc"],
  data = [":experiment_configs"],
  deps = [
    "@cxxopts",
    "//common:check",
    "//common/proto:load_from_file",
    ":experiment_config_proto",
    ":world_map_config_proto",
    ":mapped_landmarks_proto",
    "//planning:road_map_proto",
  ]
)
