
load("@pip//:requirements.bzl", "requirement")

py_library(
  name = "pyollama",
  srcs = ["pyollama.py"],
  visibility=["//visibility:public"],
  deps = [
    requirement("ollama"),
    requirement("requests"),
  ]
)

py_test(
  name = "pyollama_test",
  srcs = ["pyollama_test.py"],
  deps = [":pyollama"]
)
