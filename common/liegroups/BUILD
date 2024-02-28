
load("//common/proto:proto.bzl", "multi_proto_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")

package(default_visibility=["//visibility:public"])

cc_library(
  name = "so2",
  hdrs = ["so2.hh"],
  deps = [
    "@sophus_lie//:liegroups",
  ]
)

pybind_extension(
  name = "so2_python",
  srcs = ["so2_python.cc"],
  deps = [
    ":so2",
  ],
)

py_test(
  name = "so2_python_test",
  srcs = ["so2_python_test.py"],
  data = [
    ":so2_python.so",
  ],
  deps = [
    requirement("numpy"),
  ]
)

cc_library(
  name = "so3",
  hdrs = ["so3.hh"],
  deps = [
    "@sophus_lie//:liegroups",
  ]
)

cc_library(
  name = "se2",
  hdrs = ["se2.hh"],
  srcs = ["se2.cc"],
  deps = [
    ":so2",
    "@sophus_lie//:liegroups",
  ]
)

cc_test(
  name = "se2_test",
  srcs = ["se2_test.cc"],
  deps = [
    ":se2",
    "@com_google_googletest//:gtest_main",
  ],
)

pybind_extension(
  name = "se2_python",
  srcs = ["se2_python.cc"],
  deps = [
    ":se2",
  ],
)

py_test(
  name = "se2_python_test",
  srcs = ["se2_python_test.py"],
  data = [
    ":se2_python.so",
  ],
  deps = [
    requirement("numpy"),
  ]
)

cc_library(
  name = "se3",
  hdrs = ["se3.hh"],
  srcs = ["se3.cc"],
  deps = [
    "@sophus_lie//:liegroups",
  ]
)

cc_test(
  name = "se3_test",
  srcs = ["se3_test.cc"],
  deps = [
    ":se3",
    "@com_google_googletest//:gtest_main",
  ],
)

multi_proto_library(
  name = "so2_proto",
  srcs = ["so2.proto"],
)

multi_proto_library(
  name = "so3_proto",
  srcs = ["so3.proto"],
)

multi_proto_library(
  name = "se2_proto",
  srcs = ["se2.proto"],
  deps = [
    ":so2_proto",
    "//common/math:matrix_proto",
  ]
)

multi_proto_library(
  name = "se3_proto",
  srcs = ["se3.proto"],
  deps = [
    ":so3_proto",
    "//common/math:matrix_proto",
  ]
)

cc_library(
  name = "so2_to_proto",
  hdrs = ["so2_to_proto.hh"],
  srcs = ["so2_to_proto.cc"],
  deps = [
    ":so2",
    ":so2_proto",
  ]
)

cc_test(
  name = "so2_to_proto_test",
  srcs = ["so2_to_proto_test.cc"],
  deps = [
    ":so2_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "so3_to_proto",
  hdrs = ["so3_to_proto.hh"],
  srcs = ["so3_to_proto.cc"],
  deps = [
    ":so3",
    ":so3_proto",
  ]
)

cc_test(
  name = "so3_to_proto_test",
  srcs = ["so3_to_proto_test.cc"],
  deps = [
    ":so3_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "se2_to_proto",
  hdrs = ["se2_to_proto.hh"],
  srcs = ["se2_to_proto.cc"],
  deps = [
    ":se2",
    ":se2_proto",
    ":so2_to_proto",
    "//common/math:matrix_to_proto",
  ]
)

cc_test(
  name = "se2_to_proto_test",
  srcs = ["se2_to_proto_test.cc"],
  deps = [
    ":se2_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "se3_to_proto",
  hdrs = ["se3_to_proto.hh"],
  srcs = ["se3_to_proto.cc"],
  deps = [
    ":se3",
    ":se3_proto",
    ":so3_to_proto",
    "//common/math:matrix_to_proto",
  ]
)

cc_test(
  name = "se3_to_proto_test",
  srcs = ["se3_to_proto_test.cc"],
  deps = [
    ":se3_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)
