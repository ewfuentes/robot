package(features=["warning_compile_flags"])

load("//common/proto:proto.bzl", "multi_proto_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

cc_library(
  name = "probabilistic_road_map",
  hdrs = ["probabilistic_road_map.hh"],
  visibility = ["//visibility:public"],
  deps = [
    ":road_map",
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

pybind_extension(
  name = "probabilistic_road_map_python",
  srcs = ["probabilistic_road_map_python.cc"],
  visibility=["//visibility:public"],
  deps = [
    ":probabilistic_road_map",
    ":road_map_to_proto",
  ]
)


cc_library(
  name = "road_map",
  hdrs = ["road_map.hh"],
  srcs = ["road_map.cc"],
  deps = [
    "@eigen",
    "//common:check",
  ],
  visibility = ["//visibility:public"]
)

cc_test(
  name = "road_map_test",
  srcs = ["road_map_test.cc"],
  deps = [
    ":road_map",
    "//common:check",
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
    ":road_map",
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

cc_library(
  name = "id_a_star",
  hdrs = ["id_a_star.hh"],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "id_a_star_test",
  srcs = ["id_a_star_test.cc"],
  deps = [
    ":id_a_star",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "djikstra",
  hdrs = ["djikstra.hh"],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "djikstra_test",
  srcs = ["djikstra_test.cc"],
  deps = [
    ":djikstra",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
name = "david_planning",
hdrs = ["david_planning.hh"],
deps = [
  "//experimental/beacon_sim:ekf_slam",
  "//experimental/beacon_sim:beacon_potential",
  "//experimental/beacon_sim:belief_road_map_planner",
  "//common/math:logsumexp",
  ":djikstra",
  ":road_map"
]
)

cc_test(
  name = "david_planning_test",
  srcs = ["david_planning_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":david_planning",
    "//experimental/beacon_sim:test_helpers"
  ]
)
