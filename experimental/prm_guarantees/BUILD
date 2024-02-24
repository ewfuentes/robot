
cc_binary(
  name="annoy_precision",
  srcs = ["annoy_precision.cc"],
  deps = ["@annoy"],
)

cc_test(
  name = "nearest_neighbor_graph_test",
  srcs = ["nearest_neighbor_graph_test.cc"],
  tags = ["manual"],
  linkopts = ["-ltbb"],
  deps = [
    "@annoy",
    "@com_google_googletest//:gtest_main",
  ]
)
