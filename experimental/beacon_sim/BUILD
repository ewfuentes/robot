
cc_binary(
  name = "beacon_sim",
  srcs = ["beacon_sim.cc"],
  linkopts = ["-latomic", "-lGLU"],
  deps = [
    ":world_map",
    ":robot",
    ":generate_observations",
    ":sim_clock",
    "//visualization/gl_window",
  ],
)

cc_library(
  name = "world_map",
  hdrs = ["world_map.hh"],
  srcs = ["world_map.cc"],
  deps = [
    "@eigen//:eigen",
  ],
)

cc_library(
  name = "robot",
  hdrs = ["robot.hh"],
  srcs = ["robot.cc"],
  deps = [
    "@eigen//:eigen",
    "@sophus//:liegroups",
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
  ]
)

