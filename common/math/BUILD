
package(features=["warning_compile_flags"])

load("@//common/proto:proto.bzl", "multi_proto_library")


multi_proto_library(
  name = "matrix_proto",
  srcs = ["matrix.proto"],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "matrix_to_proto",
  hdrs = ["matrix_to_proto.hh"],
  deps = [
    "@eigen//:eigen",
    ":matrix_proto",
  ],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "matrix_to_proto_test",
  srcs = ["matrix_to_proto_test.cc"],
  deps = [
    ":matrix_to_proto",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "redheffer_star",
  hdrs = ["redheffer_star.hh"],
  srcs = ["redheffer_star.cc"],
  visibility = ["//visibility:public"],
  deps = [
    "@eigen//:eigen",
  ]
)

cc_test(
  name = "redheffer_star_test",
  srcs = ["redheffer_star_test.cc"],
  deps = [
    ":redheffer_star",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "n_choose_k",
  hdrs = ["n_choose_k.hh"],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "n_choose_k_test",
  srcs = ["n_choose_k_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    ":n_choose_k",
  ]
)

cc_library(
  name = "combinations",
  hdrs = ["combinations.hh"],
  srcs = ["combinations.cc"],
  deps = [":n_choose_k"],
  visibility = ["//visibility:public"],
)

cc_test(
  name = "combinations_test",
  srcs = ["combinations_test.cc"],
  deps = [
    ":combinations",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "sample_without_replacement",
  hdrs = ["sample_without_replacement.hh"],
  srcs = ["sample_without_replacement.cc"],
  visibility = ["//visibility:public"],
  deps = [
    "//common:argument_wrapper",
    "//common:check"
  ]
)

cc_test(
  name = "sample_without_replacement_test",
  srcs = ["sample_without_replacement_test.cc"],
  deps = [
    ":sample_without_replacement",
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_googletest//:gtest_main",
  ]
)

cc_library(
  name = "logsumexp",
  hdrs = ["logsumexp.hh"],
  visibility = ["//visibility:public"]
)

cc_library(
  name = "multivariate_normal_cdf",
  hdrs = ["multivariate_normal_cdf.hh"],
  srcs = ["multivariate_normal_cdf.cc"],
  visibility = ["//visibility:public"],
  deps = [
    "@approxcdf",
    "@eigen",
    "//common:check",
  ]
)

cc_test(
  name = "multivariate_normal_cdf_test",
  srcs = ["multivariate_normal_cdf_test.cc"],
  deps = [
    ":multivariate_normal_cdf",
    "@com_google_googletest//:gtest_main",
  ]
)
