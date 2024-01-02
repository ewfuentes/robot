
cc_library(
    name = "puzzle",
    hdrs = ["puzzle.hh"],
    srcs = ["puzzle.cc"],
    deps = [
      "//planning:id_a_star",
      "//common:check",
    ]
)

cc_binary(
    name = "polytope",
    srcs = ["polytope.cc"],
    deps = [
      ":puzzle",
      "@cxxopts",
      "@json",
      "//common:check",
    ]
)