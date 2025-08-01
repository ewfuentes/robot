load("//common/proto:proto.bzl", "multi_proto_library")
# load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
# load("@rules_python//python:defs.bzl", "py_test")
# load("@pip//:requirements.bzl", "requirement")

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
        ":frontend",
        ":frame",
        ":image_point",
        "//common:check",
        "//visualization/opencv:opencv_viz",
        "@eigen",
        "@gtsam",
        "@opencv",
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
        "//common:check",
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
        "@com_google_absl//absl/strings:str_format",
        "@nmea",
    ]
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
        ":frontend",
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
        "//common:check",
        "@gtsam",
        "@opencv",
        "@eigen"
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
        "//common:check"
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
        "//common:check",
        "@eigen",
        "@gtsam"
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
        "//common/geometry:camera",
        "@opencv",
        "@gtsam",
        "@eigen",
    ],
)

cc_test(
    name = "frontend_test",
    srcs = ["frontend_test.cc"],
    deps = [
        ":frame",
        ":frontend",
        ":image_point",
        "@com_google_googletest//:gtest_main",
        "@opencv",
        "@eigen"
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
        ":structure_from_motion_types",
        ":frame",
        "//common:check",
        "//common/geometry:camera",
        "@boost//:smart_ptr",
        "@gtsam",
        "@opencv",
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

cc_binary(
    name = "structure_from_motion_example",
    srcs = ["structure_from_motion_example.cc"],
    copts = [
        "-Wno-error=unused-parameter",
        "-Wno-error=unused-function",
    ],
    deps = [
        ":structure_from_motion",
        ":frontend",
        ":backend",
        ":four_seasons_parser",
        ":frame",
        ":image_point",
        ":image_point_four_seasons",
        ":spatial_test_scene_cube",
        ":structure_from_motion_types",
        ":camera_calibration",
        "//visualization/opencv:opencv_viz",
        "@boost//:smart_ptr",
        "@cxxopts",
        "@eigen",
        "@opencv",
    ],
)

cc_library(
    name = "translation_prior",
    hdrs = ["translation_prior.hh"],
    deps = [
        "@eigen"
    ]
)

multi_proto_library(
    name = "translation_prior_proto",
    srcs = ["translation_prior.proto"],
    deps = [
        "//common/math:matrix_proto"
    ]
)

cc_library(
    name = "translation_prior_to_proto",
    hdrs = ["translation_prior_to_proto.hh"],
    srcs = ["translation_prior_to_proto.cc"],
    deps = [
        ":translation_prior",
        ":translation_prior_proto",
        "//common/math:matrix_to_proto",
        "@eigen"
    ]
)

cc_test(
    name = "translation_prior_to_proto_test",
    srcs = ["translation_prior_to_proto_test.cc"],
    deps = [
        ":translation_prior",
        ":translation_prior_proto",
        ":translation_prior_to_proto",
        "@eigen",
        "@com_google_googletest//:gtest_main"
    ]
)

# pybind_extension(
#     name = "translation_prior_python",
#     srcs = [
#         "translation_prior_python.cc"
#     ],
#     deps = [
#         ":translation_prior"
#     ]
# )

# py_test(
#     name = "translation_prior_python_test",
#     srcs = ["translation_prior_python_test.cc"]

# )

cc_binary(
    name = "translation_prior_proto_write_example",
    srcs = ["translation_prior_proto_write_example.cc"],
    deps = [
        ":translation_prior",
        ":translation_prior_proto",
        ":translation_prior_to_proto",
        "@cxxopts",
        "@eigen"
    ]
)

cc_binary(
    name = "write_to_colmap_database",
    srcs = [
        "write_to_colmap_database.cc"
    ], 
    copts = [
        "-Wno-error=unused-parameter",
    ],
    deps = [
        ":frontend",
        ":frame",
        ":backend",
        ":image_point_four_seasons",
        ":four_seasons_parser",
        "//common/sqlite3:sqlite3",
        "//common:check",
        "@cxxopts", 
        "@eigen"
    ]
)