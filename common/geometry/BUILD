
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
  name = "camera_geometry",
  hdrs = ["camera_geometry.hh"],
  visibility = ["//visibility:public"],
  deps = [
    "@eigen",
  ]
)

cc_test(
  name = "camera_geometry_test",
  srcs = ["camera_geometry_test.cc"],
  deps = [
    ":camera_geometry",
    "@com_google_googletest//:gtest_main",
  ]
)
