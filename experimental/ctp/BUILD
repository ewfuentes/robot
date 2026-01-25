
package(features=["warning_compile_flags"])

cc_library(
  name = "compute_optimistic_rollout",
  hdrs = ["compute_optimistic_rollout.hh"],
  srcs = ["compute_optimistic_rollout.cc"],
  deps = [
    "//domain:canadian_traveler",
  ]
)

cc_test(
  name = "compute_optimistic_rollout_test",
  srcs = ["compute_optimistic_rollout_test.cc"],
  deps = [
    ":compute_optimistic_rollout",
    "@com_google_googletest//:gtest_main",
  ]
)
