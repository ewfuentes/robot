
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
  name = "opencv_viz",
  hdrs = ["opencv_viz.hh"],  
  srcs = ["opencv_viz.cc"],
  visibility = ["//visibility:public"],
  deps = [
    "@eigen",
    "@opencv//:opencv",
    ":translate_types"
  ]
)

cc_test(
  name = "opencv_viz_test",
  srcs = ["opencv_viz_test.cc"],
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

cc_library(
  name = "camera", 
  hdrs = ["camera.hh"],  
  srcs = ["camera.cc"],
  visibility = ["//visibility:public"],
  deps = [
    "@eigen",
    "@gtsam//:gtsam",
    "@opencv//:opencv",
    ":translate_types"
  ]
)

cc_test(
  name = "camera_test", 
  srcs = ["camera_test.cc"],
  deps = [
    ":camera",
    "@com_google_googletest//:gtest_main",
    ":translate_types",
    ":opencv_viz"
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