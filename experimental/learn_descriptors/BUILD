package(features=["warning_compile_flags"])

cc_library(
  name = "learn_descriptors",
  hdrs = ["learn_descriptors.hh"],
  visibility = ["//visibility:public"],
  srcs = ["learn_descriptors.cc"],
)

cc_test(
  name = "learn_descriptors_test",
  srcs = ["learn_descriptors_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":learn_descriptors",
  ]
)

cc_library(
  name = "structure_from_motion",
  hdrs = ["structure_from_motion.hh"],
  visibility = ["//visibility:public"],
  srcs = ["structure_from_motion.cc"],
  copts = [
    "-Wno-error=unused-parameter",
  ],
  deps = [
    "@opencv//:opencv",
    "@gtsam//:gtsam",
    "@eigen",
    "//common/geometry:camera",
    "//common/geometry:translate_types",
    "//common/geometry:opencv_viz",
    ":feature_manager"
  ]
)

cc_test(
  name = "structure_from_motion_test",
  srcs = ["structure_from_motion_test.cc"],  
  copts = [
    "-Wno-error=unused-parameter",
  ],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":symphony_lake_parser",
    ":structure_from_motion",
    "//common/geometry:opencv_viz",
    "//common/geometry:translate_types"
  ]
)

cc_library(
  name = "symphony_lake_parser",
  hdrs = ["symphony_lake_parser.hh"],
  copts = [
    "-Wno-unused-parameter",
  ],
  data = ["@symphony_lake_snippet"],
  visibility = ["//visibility:public"],
  srcs = ["symphony_lake_parser.cc"],
  deps = [
    "@symphony_lake_parser",
    "//common:check",
    "@eigen"
  ]
)

cc_test(
  name = "symphony_lake_parser_test",
  srcs = ["symphony_lake_parser_test.cc"],
  copts = ["-Wno-unused-parameter"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":symphony_lake_parser",
    "//common:check",
    "//common/geometry:opencv_viz"
  ]
)

cc_test(
  name = "gtsam_test",
  srcs = ["gtsam_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "@gtsam//:gtsam",
    ":learn_descriptors"  
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
  name = "feature_set",
  hdrs = ["feature_set.hh"],
  visibility = ["//visibility:public"],
  deps = [        
    "@gtsam//:gtsam",
    "@opencv//:opencv"
  ]
)

cc_library(
  name = "feature_manager",
  hdrs = ["feature_manager.hh"],
  visibility = ["//visibility:public"],
  deps = [        
    "@gtsam//:gtsam",
    "@opencv//:opencv",
    ":feature_set",
  ]
)

cc_test(
  name = "feature_manager_test",
  srcs = ["feature_manager_test.cc"],
  deps = [
    ":feature_manager",
    "@com_google_googletest//:gtest_main",
    "@gtsam//:gtsam",
    "@opencv//:opencv",
    "@eigen",
    ":spatial_scene_test_cube"
  ]
)


cc_library(
  name = "spatial_scene_test",
  hdrs = ["spatial_scene_test.hh"],
  visibility = ["//visibility:public"],
  deps = [        
    "@gtsam//:gtsam",
    "@opencv//:opencv",    
    "@eigen"
  ]
)

cc_library(
  name = "spatial_scene_test_cube",
  hdrs = ["spatial_scene_test_cube.hh"],
  visibility = ["//visibility:public"],
  deps = [        
    "@eigen",
    ":spatial_scene_test"
  ]
)