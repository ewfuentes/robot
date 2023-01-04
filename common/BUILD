
cc_library(
  name="argument_wrapper",
  hdrs=["argument_wrapper.hh"],
  visibility=["//visibility:public"],
)

cc_test(
  name = "argument_wrapper_test",
  srcs = ["argument_wrapper_test.cc"],
  deps = [
    ":argument_wrapper",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "indexed_array",
  hdrs = ["indexed_array.hh"],
  deps = ["@wise_enum//:wise_enum"],
  visibility=["//visibility:public"],
)

cc_test(
  name = "indexed_array_test",
  srcs = ["indexed_array_test.cc"],
  deps =  [
    ":indexed_array",
    "@wise_enum//:wise_enum",
    "@com_google_googletest//:gtest_main",
  ]
)