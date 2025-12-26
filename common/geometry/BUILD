load("@pip//:requirements.bzl", "requirement", "extra_requirement")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

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


cc_library(
  name = "camera",
  hdrs = ["camera.hh"],
  srcs = ["camera.cc"],
  visibility = ["//visibility:public"],
  deps = [
    "@eigen",     
    "@opencv",
    ":translate_types",
    "@gtsam",
    "//common:check"
  ]
)

cc_test(
  name = "camera_test",
  srcs = ["camera_test.cc"],
  visibility = ["//visibility:public"],
  deps = [
    ":camera",
    "@com_google_googletest//:gtest_main",
    "//visualization/opencv:opencv_viz",
    ":translate_types"
  ]
)

# Python GPU geometry library
py_library(
    name = "gpu_distance",
    srcs = ["gpu_distance.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//common/torch:load_torch_deps",
        requirement("torch"),
    ],
)

py_test(
    name = "gpu_distance_test",
    srcs = ["gpu_distance_test.py"],
    deps = [
        ":gpu_distance",
        "//common/torch:load_torch_deps",
        requirement("numpy"),
        requirement("shapely"),
    ],
)

py_library(
    name = "gpu_geometry_collection",
    srcs = ["gpu_geometry_collection.py"],
    data = [":spatial_distance_python.so"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_distance",
        requirement("shapely"),
        requirement("torch"),
    ],
)

py_test(
    name = "gpu_geometry_collection_test",
    srcs = ["gpu_geometry_collection_test.py"],
    data = ["@vigor_snippet//:files"],
    deps = [
        ":gpu_geometry_collection",
        "//common/torch:load_torch_deps",
        requirement("geopandas"),
        requirement("pyarrow"),
        requirement("shapely"),
    ],
)

py_test(
    name = "spatial_distance_test",
    srcs = ["spatial_distance_test.py"],
    data = [":spatial_distance_python.so"],
    deps = [
        ":gpu_geometry_collection",
        "//common/torch:load_torch_deps",
        requirement("numpy"),
        requirement("shapely"),
        requirement("torch"),
    ],
)

# CUDA-accelerated spatial distance queries
cuda_library(
    name = "spatial_distance_cuda",
    hdrs = ["spatial_distance.hh"],
    srcs = ["spatial_distance.cu"],
    deps = [
        "@pip_torch//:libtorch",
        "@fmt",
    ],
)

pybind_extension(
    name = "spatial_distance_python",
    srcs = ["spatial_distance_python.cc"],
    deps = [
        ":spatial_distance_cuda",
        extra_requirement("torch", "libtorch"),
    ],
)
