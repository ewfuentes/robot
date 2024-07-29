
package(features=["warning_compile_flags"])

cc_library(
  name = "sqlite3",
  hdrs = ["sqlite3.hh"],
  srcs = ["sqlite3.cc"],
  deps = [
    "@sqlite3",
    "//common:check",
    "//common:argument_wrapper",
  ],
)

cc_test(
  name = "sqlite3_test",
  srcs = ["sqlite3_test.cc"],
  deps = [
    ":sqlite3",
    "@com_google_googletest//:gtest_main",
  ]
)
