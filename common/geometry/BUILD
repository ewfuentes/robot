
package(features=["warning_compile_flags"])

cc_library(
  name = "nearest_point_on_segment",
  hdrs = ["nearest_point_on_segment.hh"],
  visibility = ["//visibility:public"],
  deps = [
    "@eigen",
  ]
)

cc_test(
  name = "nearest_point_on_segment_test",
  srcs = ["nearest_point_on_segment_test.cc"],
  deps = [
    ":nearest_point_on_segment",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "translate_types",
  hdrs = ["translate_types.hh"],
  srcs = ["translate_types.cc"],
  visibility = ["//visibility:public"],
  deps = [
    "@eigen",     
    "@opencv//:opencv"
  ]
)

cc_test(
  name = "translate_types_test", 
  srcs = ["translate_types_test.cc"],
  visibility = ["//visibility:public"],
  deps = [
    ":translate_types",
    "@com_google_googletest//:gtest_main"
  ]
)