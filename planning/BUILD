load("//common/proto:proto.bzl", "multi_proto_library")

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
  visibility = ["//visibility:public"],
  deps = [
    ":breadth_first_search",
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

cc_library(
  name = "breadth_first_search",
  hdrs = ["breadth_first_search.hh"],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "breadth_first_search_test",
  srcs = ["breadth_first_search_test.cc"],
  deps = [
    ":breadth_first_search",
    "@com_google_googletest//:gtest_main",
  ]
)

multi_proto_library(
  name = "road_map_proto",
  srcs = ["road_map.proto"],
  visibility = ["//visibility:public"],
  deps = [
    "//common/math:matrix_proto",
  ]
)

cc_library(
  name = "road_map_to_proto",
  hdrs = ["road_map_to_proto.hh"],
  srcs = ["road_map_to_proto.cc"],
  visibility = ["//visibility:public"],
  deps = [
    ":probabilistic_road_map",
    ":road_map_proto",
    "//common/math:matrix_to_proto",
  ]
)

cc_test(
  name = "road_map_to_proto_test",
  srcs = ["road_map_to_proto_test.cc"],
  deps = [
    ":road_map_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)