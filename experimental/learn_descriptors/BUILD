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
    "//visualization/opencv:opencv_viz",
    ":feature_manager",
    ":frontend",
    ":backend"
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
    "//visualization/opencv:opencv_viz",
    "//common/geometry:translate_types",
    ":frame"
  ]
)

cc_library(
  name = "symphony_lake_parser",
  hdrs = ["symphony_lake_parser.hh"],
  copts = [
    "-Wno-unused-parameter",
  ],
  data = ["@symphony_lake_snippet//:files"],
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
  data = ["@symphony_lake_snippet//:files"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":symphony_lake_parser",
    "//visualization/opencv:opencv_viz"
  ]
)

cc_test(
  name = "gtsam_test",
  srcs = ["gtsam_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "@gtsam//:gtsam",
  ]
)

cc_library(
    name = "image_point", 
    hdrs = ["image_point.hh"],
    deps = [
      "//common/liegroups:se3",
      "@opencv//:opencv",
      "@eigen"
    ]
)

cc_library(
    name = "four_seasons_parser", 
    hdrs = ["four_seasons_parser.hh"],
    srcs = ["four_seasons_parser.cc"],
    deps = [
      "@eigen",
      "@opencv//:opencv",
      "//common/liegroups:se3",
      "//common/time:robot_time",
      "@com_google_absl//absl/strings:str_format",
      ":image_point",
      "@nmea",
      "@geographiclib",
      "//common:check"
    ]
)

cc_binary(
  name = "four_seasons_parser_example",
  srcs = ["four_seasons_parser_example.cc"],
  deps = [
    ":four_seasons_parser",
    "@cxxopts//:cxxopts",
    "//common:check",
    "@geographiclib"
  ]
)

cc_binary(
  name = "four_seasons_parser_example_viz",
  srcs = ["four_seasons_parser_example_viz.cc"],
  deps = [
    ":four_seasons_parser",
    "@cxxopts//:cxxopts",
    "//common:check",
    "//visualization/opencv:opencv_viz",
    "@eigen",    
  ]
)

cc_test(
  name = "four_seasons_parser_test",
  srcs = ["four_seasons_parser_test.cc"],
  data = ["@four_seasons_snippet//:files"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "//common/liegroups:se3",
    "@com_google_absl//absl/strings:str_format",
    "@eigen",
    ":four_seasons_parser",
    "@nmea",
    "@geographiclib",
    "//common:check"
  ]
)

cc_library(
  name = "frontend_definitions",
  hdrs = ["frontend_definitions.hh"],
  visibility = ["//visibility:public"],
  deps = [        
    "@gtsam//:gtsam",    
    ":structure_from_motion_types"
  ]
)

cc_library(
  name = "structure_from_motion_types",
  hdrs = ["structure_from_motion_types.hh"],
  visibility = ["//visibility:public"],
  deps = [        
    "@gtsam//:gtsam",    
    "@opencv//:opencv"
  ]
)

cc_library(
  name = "frame",
  hdrs = ["frame.hh"],
  srcs = ["frame.cc"],
  visibility = ["//visibility:public"],
  deps = [             
    "@opencv//:opencv",
    ":frontend_definitions",
    "@gtsam//:gtsam"
  ]
)

cc_test(
  name = "sfm_mvp",
  srcs = ["sfm_mvp.cc"],
    copts = [
    "-Wno-error=unused-parameter",
    "-Wno-error=unused-variable",
  ],
  deps = [
    ":frontend",
    "@com_google_googletest//:gtest_main",
    "@gtsam//:gtsam",
    "@opencv//:opencv",
    "@eigen",
    ":spatial_test_scene_cube",
    ":symphony_lake_parser",
    ":frame", 
    ":structure_from_motion_types",
    "//visualization/opencv:opencv_viz"
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
    ":spatial_test_scene_cube"
  ]
)


cc_library(
  name = "spatial_test_scene",
  hdrs = ["spatial_test_scene.hh"],
  visibility = ["//visibility:public"],
  srcs = ["spatial_test_scene.cc"],
  deps = [        
    "@gtsam//:gtsam",
    "@opencv//:opencv",    
    "@eigen"
  ]
)

cc_library(
  name = "spatial_test_scene_cube",
  hdrs = ["spatial_test_scene_cube.hh"],
  visibility = ["//visibility:public"],
  deps = [        
    "@eigen",
    ":spatial_test_scene"
  ]
)

cc_binary(
  name = "spatial_test_scene_cube_example",
  srcs = ["spatial_test_scene_cube_example.cc"],
  deps = [    
    "@com_google_googletest//:gtest_main",    
    ":spatial_test_scene_cube",
    "//visualization/opencv:opencv_viz",
    "@eigen"
  ]
)

cc_library(
  name = "frontend",
  hdrs = ["frontend.hh"],
  visibility = ["//visibility:public"],
  srcs = ["frontend.cc"],
  copts = [
    "-Wno-error=unused-parameter",
  ],
  deps = [
    "@opencv//:opencv",    
    ":frame",
    ":frontend_definitions",
    ":structure_from_motion_types"
  ]
)


cc_test(
  name = "frontend_test",
  srcs = ["frontend_test.cc"],
  deps = [    
    "@com_google_googletest//:gtest_main",    
    ":frontend"
  ]
)

cc_library(
  name = "backend",
  hdrs = ["backend.hh"],
  visibility = ["//visibility:public"],
  srcs = ["backend.cc"],
  copts = [
    "-Wno-error=unused-parameter",
  ],
  deps = [
    "@opencv//:opencv",    
    "@gtsam//:gtsam",
    "//common/geometry:camera",
    "@boost//:smart_ptr",
    ":feature_manager"
  ]
)


cc_test(
  name = "backend_test",
  srcs = ["backend_test.cc"],
    copts = [
    "-Wno-error=unused-parameter",
  ],
  deps = [    
    "@com_google_googletest//:gtest_main",    
    "@opencv//:opencv",
    ":backend",
    ":feature_manager",
    ":spatial_test_scene_cube",
    "//visualization/opencv:opencv_viz"
  ]
)

cc_binary(
  name = "get_colmap_groundtruth",
  srcs = ["get_colmap_groundtruth.cc"],
  copts = [
    "-Wno-error=unused-parameter",
  ],
  deps = [
    ":symphony_lake_parser"
  ]
)

cc_binary(
  name = "incremental_sfm_mvp",
  srcs = ["incremental_sfm_mvp.cc"],
  copts = [
    "-Wno-error=unused-parameter",
    "-Wno-error=unused-function",
  ],
  deps = [
    ":four_seasons_parser",
    "@eigen",
    "@opencv",    
    "@gtsam",
    "//visualization/opencv:opencv_viz",
    ":spatial_test_scene_cube",
    ":frame", 
    ":structure_from_motion_types",
    ":frontend",
    "@boost//:smart_ptr",    
    "@cxxopts",
    "//common/geometry:camera"
  ]
)
