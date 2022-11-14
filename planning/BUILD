
cc_library(
  name = "probabilistic_road_map",
  hdrs = ["probabilistic_road_map.hh"],
  visibility = ["//visibility:public"],
  deps = [
    "@eigen//:eigen",
  ]
)

cc_test(
  name = "probabilistic_road_map_test",
  srcs = ["probabilistic_road_map_test.cc"],
  deps = [
    ":probabilistic_road_map",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "a_star",
  hdrs = ["a_star.hh"],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "a_star_test",
  srcs = ["a_star_test.cc"],
  deps = [
    ":a_star",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "belief_road_map",
  hdrs = ["belief_road_map.hh"],
  srcs = ["belief_road_map.cc"],
  deps = [
    ":a_star",
    ":probabilistic_road_map",
    "@eigen//:eigen",
  ],
)

cc_test(
  name = "belief_road_map_test",
  srcs = ["belief_road_map_test.cc"],
  deps = [
    ":belief_road_map",
    "@com_google_googletest//:gtest_main",
  ],
)

