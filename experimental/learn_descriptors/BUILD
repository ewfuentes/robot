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
    name = "visual_odometry",
    srcs = ["visual_odometry.cc"],
    hdrs = ["visual_odometry.hh"],
    visibility = ["//visibility:public"],
    deps = [
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
    name = "visual_odometry_test",
    srcs = ["visual_odometry_test.cc"],
    deps = [
        ":visual_odometry",
        "@com_google_googletest//:gtest_main",
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
    srcs = ["four_seasons_transforms.cc"],
    hdrs = ["four_seasons_transforms.hh"],
    visibility = ["//visibility:public"],
    deps = [
        ":four_seasons_parser_detail",
        "//common:check",
        "//common/liegroups:se3",
        "@eigen",
    ],
)

cc_library(
    name = "image_point_four_seasons",
    srcs = ["image_point_four_seasons.cc"],
    hdrs = ["image_point_four_seasons.hh"],
    visibility = ["//visibility:public"],
    deps = [
        ":four_seasons_transforms",
        ":gps_data",
        ":image_point",
        "//common:check",
        "//common/gps:frame_translation",
        "//common/liegroups:se3",
        "@eigen",
        "@geographiclib",
        "@opencv",
    ],
)

cc_library(
    name = "gps_data",
    hdrs = ["gps_data.hh"],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen",
    ],
)

cc_library(
    name = "four_seasons_parser",
    srcs = ["four_seasons_parser.cc"],
    hdrs = ["four_seasons_parser.hh"],
    visibility = ["//visibility:public"],
    deps = [
        ":camera_calibration",
        ":four_seasons_parser_detail",
        ":four_seasons_transforms",
        ":image_point_four_seasons",
        "//common:check",
        "//common/liegroups:se3",
        "@eigen",
        "@opencv",
    ],
)

cc_library(
    name = "four_seasons_parser_detail",
    srcs = ["four_seasons_parser_detail.cc"],
    hdrs = ["four_seasons_parser_detail.hh"],
    visibility = ["//visibility:public"],
    deps = [
        ":camera_calibration",
        ":gps_data",
        "@com_google_absl//absl/strings:str_format",
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

cc_test(
    name = "four_seasons_parser_test",
    srcs = ["four_seasons_parser_test.cc"],
    data = ["@four_seasons_snippet//:files"],
    deps = [
        ":camera_calibration",
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
    ],
)

cc_library(
    name = "camera_calibration",
    hdrs = ["camera_calibration.hh"],
    visibility = ["//visibility:public"],
    deps = [
        "@opencv",
    ],
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
        ":image_point_four_seasons",
        ":four_seasons_parser",
        "//common/sqlite3:sqlite3",
        "//common:check",
        "@cxxopts", 
        "@eigen"
    ]
)

py_binary(
    name = "truncate_4seasons",
    srcs = ["truncate_4seasons.py"],
)
