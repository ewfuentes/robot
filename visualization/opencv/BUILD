package(features=["warning_compile_flags"])

cc_library(
  name = "opencv_viz",
  hdrs = ["opencv_viz.hh"],  
  srcs = ["opencv_viz.cc"],
  visibility = ["//visibility:public"],
  deps = [
    "@eigen",
    "@opencv//:opencv",
    "//common/geometry:translate_types"
  ]
)

cc_test(
  name = "opencv_viz_test",
  srcs = ["opencv_viz_test.cc"],
  tags = ["manual"],
  deps = [
    ":opencv_viz",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_binary(
  name = "opencv_viz_example",
  srcs = ["opencv_viz_example.cc"],
  deps = [
    "@opencv//:opencv"
  ]
)