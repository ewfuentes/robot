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
<<<<<<< HEAD
=======
cc_library(
  name = "vo",
  hdrs = ["vo.hh"],
  visibility = ["//visibility:public"],
  srcs = ["vo.cc"],
  deps = [
    "@opencv//:opencv",
  ]
)

cc_test(
  name = "vo_test",
  srcs = ["vo_test.cc"],  
  deps = [
    "@com_google_googletest//:gtest_main",
    ":vo"
  ]
)
>>>>>>> 6b172b3 (Frontend with various algorithms (SIFT, ORB, different matching algorithms) done)
