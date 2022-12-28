
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
  deps = [":fog"],
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