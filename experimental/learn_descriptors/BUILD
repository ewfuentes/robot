package(features = ["warning_compile_flags"])

cc_library(
    name = "learn_descriptors",
    srcs = ["learn_descriptors.cc"],
    hdrs = ["learn_descriptors.hh"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "learn_descriptors_test",
    srcs = ["learn_descriptors_test.cc"],
    deps = [
        ":learn_descriptors",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "structure_from_motion",
    srcs = ["structure_from_motion.cc"],
    hdrs = ["structure_from_motion.hh"],
    copts = [
        "-Wno-error=unused-parameter",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":backend",
        ":feature_manager",
        ":frontend",
        "//visualization/opencv:opencv_viz",
        "@eigen",
        "@gtsam",
        "@opencv",
    ],
)

cc_test(
    name = "structure_from_motion_test",
    srcs = ["structure_from_motion_test.cc"],
    copts = [
        "-Wno-error=unused-parameter",
    ],
    deps = [
        ":frame",
        ":structure_from_motion",
        ":symphony_lake_parser",
        "//common/geometry:translate_types",
        "//visualization/opencv:opencv_viz",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "symphony_lake_parser",
    srcs = ["symphony_lake_parser.cc"],
    hdrs = ["symphony_lake_parser.hh"],
    copts = [
        "-Wno-unused-parameter",
    ],
    data = ["@symphony_lake_snippet//:files"],
    visibility = ["//visibility:public"],
    deps = [
        "//common:check",
        "@eigen",
        "@symphony_lake_parser",
    ],
)

cc_test(
    name = "symphony_lake_parser_test",
    srcs = ["symphony_lake_parser_test.cc"],
    copts = ["-Wno-unused-parameter"],
    data = ["@symphony_lake_snippet//:files"],
    deps = [
        ":symphony_lake_parser",
        "//visualization/opencv:opencv_viz",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gtsam_test",
    srcs = ["gtsam_test.cc"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@gtsam",
    ],
)

cc_library(
    name = "image_point",
    hdrs = ["image_point.hh"],
    visibility = ["//visibility:public"],
    deps = [
        "//common/liegroups:se3",
        "@eigen",
    ],
)

cc_library(
    name = "four_seasons_transforms",
    hdrs = ["four_seasons_transforms.hh"],
    srcs = ["four_seasons_transforms.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":four_seasons_parser_detail",
        "//common/liegroups:se3",
        "//common:check",
        "@eigen",
    ],
)

cc_library(
    name = "image_point_four_seasons",
    hdrs = ["image_point_four_seasons.hh"],
    srcs = ["image_point_four_seasons.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":image_point",
        ":gps_data",
        ":four_seasons_transforms",
        "//common/liegroups:se3",
        "//common/gps:frame_translation",
        "@eigen",
        "@opencv",
        "@geographiclib",
    ]
)

cc_library(
    name = "gps_data",
    hdrs = ["gps_data.hh"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
    ]
)

cc_library(
    name = "four_seasons_parser",
    hdrs = ["four_seasons_parser.hh"],
    srcs = ["four_seasons_parser.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":image_point_four_seasons",
        ":four_seasons_transforms",
        ":four_seasons_parser_detail",
        "//common:check",
        "//common/liegroups:se3",
        "@eigen",
        "@opencv",
        ":camera_calibration"
    ],
)

cc_library(
    name = "four_seasons_parser_detail",
    hdrs = ["four_seasons_parser_detail.hh"],
    srcs = ["four_seasons_parser_detail.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":camera_calibration",
        ":gps_data",
        "//common/liegroups:se3",
        "@com_google_absl//absl/strings:str_format",
        "@geographiclib",
        "@nmea",
    ],
)

cc_binary(
    name = "four_seasons_parser_example",
    srcs = ["four_seasons_parser_example.cc"],
    deps = [
        ":four_seasons_parser",
        ":gps_data",
        "//common:check",
        "@cxxopts",
    ],
)

cc_binary(
    name = "four_seasons_parser_example_viz",
    srcs = ["four_seasons_parser_example_viz.cc"],
    deps = [
        ":four_seasons_parser",
        "//common:check",
        "//common/gps:frame_translation",
        "//visualization/opencv:opencv_viz",
        "@cxxopts",
        "@eigen",
    ],
)

cc_test(
    name = "four_seasons_parser_test",
    srcs = ["four_seasons_parser_test.cc"],
    data = ["@four_seasons_snippet//:files"],
    deps = [
        ":four_seasons_parser",
        ":four_seasons_parser_detail",
        ":image_point_four_seasons",
        "//common:check",
        "//common/gps:frame_translation",
        "//common/liegroups:se3",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@eigen",
        "@geographiclib",
        "@nmea",
        ":camera_calibration"
    ],
)

cc_library(
    name = "frontend_definitions",
    hdrs = ["frontend_definitions.hh"],
    visibility = ["//visibility:public"],
    deps = [
        ":structure_from_motion_types",
        "@gtsam",
    ],
)

cc_library(
    name = "structure_from_motion_types",
    hdrs = ["structure_from_motion_types.hh"],
    visibility = ["//visibility:public"],
    deps = [
        "@gtsam",
        "@opencv",
    ],
)

cc_library(
    name = "frame",
    srcs = ["frame.cc"],
    hdrs = ["frame.hh"],
    visibility = ["//visibility:public"],
    deps = [
        ":frontend_definitions",
        "@gtsam",
        "@opencv",
    ],
)

cc_test(
    name = "sfm_mvp",
    srcs = ["sfm_mvp.cc"],
    copts = [
        "-Wno-error=unused-parameter",
        "-Wno-error=unused-variable",
    ],
    deps = [
        ":frame",
        ":frontend",
        ":spatial_test_scene_cube",
        ":structure_from_motion_types",
        ":symphony_lake_parser",
        "//visualization/opencv:opencv_viz",
        "@com_google_googletest//:gtest_main",
        "@eigen",
        "@gtsam",
        "@opencv",
    ],
)

cc_library(
    name = "feature_set",
    hdrs = ["feature_set.hh"],
    visibility = ["//visibility:public"],
    deps = [
        "@gtsam",
        "@opencv",
    ],
)

cc_library(
    name = "feature_manager",
    hdrs = ["feature_manager.hh"],
    visibility = ["//visibility:public"],
    deps = [
        ":feature_set",
        "@gtsam",
        "@opencv",
    ],
)

cc_test(
    name = "feature_manager_test",
    srcs = ["feature_manager_test.cc"],
    deps = [
        ":feature_manager",
        ":spatial_test_scene_cube",
        "@com_google_googletest//:gtest_main",
        "@eigen",
        "@gtsam",
        "@opencv",
    ],
)

cc_library(
    name = "spatial_test_scene",
    srcs = ["spatial_test_scene.cc"],
    hdrs = ["spatial_test_scene.hh"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
        "@gtsam",
        "@opencv",
    ],
)

cc_library(
    name = "spatial_test_scene_cube",
    hdrs = ["spatial_test_scene_cube.hh"],
    visibility = ["//visibility:public"],
    deps = [
        ":spatial_test_scene",
        "@eigen",
    ],
)

cc_binary(
    name = "spatial_test_scene_cube_example",
    srcs = ["spatial_test_scene_cube_example.cc"],
    deps = [
        ":spatial_test_scene_cube",
        "//visualization/opencv:opencv_viz",
        "@com_google_googletest//:gtest_main",
        "@eigen",
    ],
)

cc_library(
    name = "frontend",
    srcs = ["frontend.cc"],
    hdrs = ["frontend.hh"],
    copts = [
        "-Wno-error=unused-parameter",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":frame",
        ":frontend_definitions",
        ":image_point_four_seasons",
        ":structure_from_motion_types",
        "//common:check",
        "@opencv",
        "@gtsam",
        "@eigen",
    ],
)

cc_test(
    name = "frontend_test",
    srcs = ["frontend_test.cc"],
    deps = [
        ":frontend",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "backend",
    srcs = ["backend.cc"],
    hdrs = ["backend.hh"],
    copts = [
        "-Wno-error=unused-parameter",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":feature_manager",
        "//common/geometry:camera",
        "@boost//:smart_ptr",
        "@gtsam",
        "@opencv",
    ],
)

cc_test(
    name = "backend_test",
    srcs = ["backend_test.cc"],
    copts = [
        "-Wno-error=unused-parameter",
    ],
    deps = [
        ":backend",
        ":feature_manager",
        ":spatial_test_scene_cube",
        "//visualization/opencv:opencv_viz",
        "@com_google_googletest//:gtest_main",
        "@opencv",
    ],
)

cc_binary(
    name = "get_colmap_groundtruth",
    srcs = ["get_colmap_groundtruth.cc"],
    copts = [
        "-Wno-error=unused-parameter",
    ],
    deps = [
        ":symphony_lake_parser",
    ],
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
        ":frame",
        ":frontend",
        ":spatial_test_scene_cube",
        ":structure_from_motion_types",
        "//common/geometry:camera",
        "//visualization/opencv:opencv_viz",
        "@boost//:smart_ptr",
        "@cxxopts",
        "@eigen",
        "@gtsam",
        "@opencv",
        ":camera_calibration"
    ],
)

cc_library(
    name = "camera_calibration",
    hdrs = ["camera_calibration.hh"],
    visibility = ["//visibility:public"],
    deps = [
        "@opencv"
    ]
)