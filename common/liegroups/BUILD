
package(default_visibility=["//visibility:public"])

cc_library(
  name = "se2",
  hdrs = ["se2.hh"],
  srcs = ["se2.cc"],
  deps = [
    "@sophus//:liegroups",
  ]
)

cc_test(
  name = "se2_test",
  srcs = ["se2_test.cc"],
  deps = [
    ":se2",
    "@com_google_googletest//:gtest_main",
  ],
)

cc_library(
  name = "se3",
  hdrs = ["se3.hh"],
  srcs = ["se3.cc"],
  deps = [
    "@sophus//:liegroups",
  ]
)

cc_test(
  name = "se3_test",
  srcs = ["se3_test.cc"],
  deps = [
    ":se3",
    "@com_google_googletest//:gtest_main",
  ],
)