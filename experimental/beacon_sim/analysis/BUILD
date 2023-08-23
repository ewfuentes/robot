
load("@pip//:requirements.bzl", "requirement")

py_library(
   name = "star",
   srcs = ["star.py"],
   deps = [
     requirement("click"),
     requirement("numpy"),
   ]
)

py_library(
    name = "models",
    srcs = ["models.py"],
    deps = [requirement("numpy")]
)

py_library(
    name = "belief_propagation",
    srcs = ["belief_propagation.py"],
    deps = [
      ":star"
      requirement("numpy"),
    ]
)

py_binary(
    name = "analysis",
    srcs = ["analysis.py"],
    deps = [
      ":belief_propagation",
      ":models",
      ":star",
      requirement("click"),
      requirement("matplotlib"),
      requirement("numpy"),
    ]
)
