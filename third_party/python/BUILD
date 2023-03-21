
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

compile_pip_requirements( name = "requirements",
  requirements_in="requirements.in",
  requirements_txt="requirements.txt",
  timeout="moderate",
)


py_binary(
  name = "hello_world",
  srcs = ["hello_world.py"],
  deps = [
    requirement("numpy"),
  ]
)
