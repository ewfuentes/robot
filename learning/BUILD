
cc_library(
  name = "cfr",
  hdrs = ["cfr.hh"],
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
