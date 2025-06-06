
package(features=["warning_compile_flags"])

load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@//common/python:embedded_py.bzl", "cc_py_library", "cc_py_test")

cc_library(
  name="argument_wrapper",
  hdrs=["argument_wrapper.hh"],
  visibility=["//visibility:public"],
)

cc_test(
  name = "argument_wrapper_test",
  srcs = ["argument_wrapper_test.cc"],
  deps = [
    ":argument_wrapper",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "indexed_array",
  hdrs = ["indexed_array.hh"],
  deps = ["@wise_enum//:wise_enum"],
  visibility=["//visibility:public"],
)

cc_test(
  name = "indexed_array_test",
  srcs = ["indexed_array_test.cc"],
  deps =  [
    ":indexed_array",
    "@wise_enum//:wise_enum",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "drake",
  deps = select({
    "@//toolchain:jammy_22.04": ["@drake_lib_jammy//:drake_shared_library"],
    "@//toolchain:noble_24.04": ["@drake_lib_noble//:drake_shared_library"],
  }),
  visibility = ["//visibility:public"],
)

cc_library(
  name = "check",
  hdrs = ["check.hh"],
  deps = [
    "@assert",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "check_test",
  srcs = ["check_test.cc"],
  deps = [
    ":check",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_py_library(
  name = "matplotlib",
  hdrs = ["matplotlib.hh"],
  srcs = ["matplotlib.cc"],
  visibility= ["//visibility:public"],
  py_deps = [
    requirement("matplotlib"),
    requirement("PyGObject"),
  ],
  deps = [
    "@pybind11",
    "@rules_python//python/cc:current_py_cc_headers",
    "@rules_python//python/cc:current_py_cc_libs",
  ],
)

cc_test(
  name = "matplotlib_test",
  srcs = ["matplotlib_test.cc"],
  tags = ["manual"],
  deps = [
    ":matplotlib",
    "@com_google_googletest//:gtest_main",
  ],
)

cc_test(
  name = "thread_pool_test",
  srcs = ["thread_pool_test.cc"],
  deps = [
    "@bs_thread_pool",
    "@com_google_googletest//:gtest_main",
  ],
)
