package(features=["warning_compile_flags"])

cc_library(
  name = "canadian_traveler",
  hdrs = ["canadian_traveler.hh"],
  srcs = ["canadian_traveler.cc"],
  deps = ["@eigen"],
)

cc_test(
  name = "canadian_traveler_test",
  srcs = ["canadian_traveler_test.cc"],
  deps = [
    ":canadian_traveler",
    "@com_google_googletest//:gtest_main",
  ],
)

cc_library(
  name = "tiger",
  hdrs = ["tiger.hh"],
  srcs = ["tiger.cc"],
)

cc_test(
  name = "tiger_test",
  srcs = ["tiger_test.cc"],
  deps = [
    ":tiger",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "fog",
  srcs = ["fog.hh"],
)

cc_library(
  name = "rock_paper_scissors",
  hdrs = ["rock_paper_scissors.hh"],
  srcs = ["rock_paper_scissors.cc"],
  visibility = ["//visibility:public"],
  deps = [
    ":fog",
    "@wise_enum//:wise_enum",
  ],
)

cc_binary(
  name = "rock_paper_scissors_main",
  srcs = ["rock_paper_scissors_main.cc"],
  deps = [
    ":rock_paper_scissors"
  ],
)

cc_test(
  name = "rock_paper_scissors_test",
  srcs = ["rock_paper_scissors_test.cc"],
  deps = [
    ":rock_paper_scissors",
    "@com_google_googletest//:gtest_main"
  ],
)

cc_library(
  name = "blotto",
  hdrs = ["blotto.hh"],
  srcs = ["blotto.cc"],
  visibility = ["//visibility:public"],
  deps = [
    ":fog",
    "@wise_enum//:wise_enum",
  ]
)

cc_library(
  name = "kuhn_poker",
  hdrs = ["kuhn_poker.hh"],
  srcs = ["kuhn_poker.cc"],
  visibility = ["//visibility:public"],
  deps = [
    ":fog",
    "//common:argument_wrapper",
    "//common:indexed_array",
    "@wise_enum//:wise_enum",
  ]
)

cc_test(
  name = "kuhn_poker_test",
  srcs = ["kuhn_poker_test.cc"],
  deps = [
    ":kuhn_poker",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "deck",
  hdrs = ["deck.hh"],
  visibility = ["//visibility:public"],
  deps = [
    "@wise_enum//:wise_enum",
    "@ompeval//:ompeval_lib",
    "//common:argument_wrapper",
  ],
)

cc_test(
  name = "deck_test",
  srcs = ["deck_test.cc"],
  deps = [
    ":deck",
    "@com_google_googletest//:gtest_main",
    "@com_google_absl//absl/container:flat_hash_map",
  ]
)

cc_library(
  name = "rob_poker",
  hdrs = ["rob_poker.hh"],
  srcs = ["rob_poker.cc"],
  visibility = ["//visibility:public"],
  deps = [
    ":fog",
    ":deck",
    "//common:argument_wrapper",
    "//common:indexed_array",
    "@ompeval//:ompeval_lib",
    "@wise_enum//:wise_enum",
    "//common/time:robot_time",
    "@com_google_absl//absl/container:inlined_vector",
  ]
)

cc_test(
  name = "rob_poker_test",
  srcs = ["rob_poker_test.cc"],
  deps = [
    ":rob_poker",
    "@com_google_googletest//:gtest_main",
  ]
)
