
package(features=["warning_compile_flags"])

cc_test(
  name="rosbag_reader_test",
  srcs=["rosbag_reader_test.cc"],
  tags=["manual"],
  data=["test.bag"],
  deps=[
    "@com_google_googletest//:gtest_main",
    "@embag//:embag",
  ]
)
