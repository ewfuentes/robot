package(features=["warning_compile_flags"])

load("//common/proto:proto.bzl", "multi_proto_library")

cc_library(
  name = "cfr",
  hdrs = ["cfr.hh"],
  visibility = ["//visibility:public"],
  deps = [
    "//common:argument_wrapper",
    "//common:indexed_array",
    "@wise_enum//:wise_enum",
  ]
)

cc_test(
  name = "cfr_test",
  srcs = ["cfr_test.cc"],
  deps = [
    ":cfr",
    "@com_google_googletest//:gtest_main",
    "//domain:rock_paper_scissors",
    "//domain:blotto",
    "//domain:kuhn_poker",
  ]
)

multi_proto_library(
  name = "min_regret_strategy_proto",
  visibility = ["//visibility:public"],
  srcs = ["min_regret_strategy.proto"]
)

cc_library(
  name = "min_regret_strategy_to_proto",
  visibility = ["//visibility:public"],
  hdrs = ["min_regret_strategy_to_proto.hh"],
  deps = [
      ":cfr",
      ":min_regret_strategy_proto",
  ]
)

cc_test(
  name = "min_regret_strategy_to_proto_test",
  srcs = ["min_regret_strategy_to_proto_test.cc"],
  deps = [
      ":min_regret_strategy_to_proto",
      "//domain:rock_paper_scissors",
      "@com_google_googletest//:gtest_main",
  ]
)
