
cc_library(
    name = "puzzle",
    hdrs = ["puzzle.hh"],
    srcs = ["puzzle.cc"],
    deps = [
      "//planning:a_star",
    ]
)

cc_binary(
    name = "polytope",
    srcs = ["polytope.cc"],
    deps = [
      ":puzzle",
      "@cxxopts",
      "@json",
    ]
)