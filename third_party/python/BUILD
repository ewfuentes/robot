
load("@python//3.8:defs.bzl", compile_pip_requirements_3_8 = "compile_pip_requirements")
load("@python//3.10:defs.bzl", compile_pip_requirements_3_10 = "compile_pip_requirements")
load("@python//3.12:defs.bzl", compile_pip_requirements_3_12 = "compile_pip_requirements")
load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

compile_pip_requirements_3_8(
  name = "requirements_3_8",
  requirements_in=":requirements_3_8.in",
  requirements_txt=":requirements_3_8.txt",
  extra_args=["--allow-unsafe"],
  timeout="eternal",
)

compile_pip_requirements_3_10(
  name = "requirements_3_10",
  requirements_in=":requirements_3_10.in",
  requirements_txt=":requirements_3_10.txt",
  extra_args=["--allow-unsafe"],
  timeout="eternal",
)

compile_pip_requirements_3_12(
  name = "requirements_3_12",
  requirements_in=":requirements_3_12.in",
  requirements_txt=":requirements_3_12.txt",
  extra_args=["--allow-unsafe"],
  timeout="eternal",
)

py_binary(
  name = "hello_world",
  srcs = ["hello_world.py"],
  deps = [
    requirement("numpy"),
  ]
)
