load("//third_party/python:uv_compile_pip_requirements.bzl", "uv_compile_pip_requirements")
load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

uv_compile_pip_requirements(
    name = "requirements_3_8",
    requirements_in = ":requirements_3_8.in",
    requirements_txt = ":requirements_3_8.txt",
    python_version = "3_8",
)

uv_compile_pip_requirements(
    name = "requirements_3_10",
    requirements_in = ":requirements_3_10.in",
    requirements_txt = ":requirements_3_10.txt",
    python_version = "3_10",
)

uv_compile_pip_requirements(
    name = "requirements_3_12",
    requirements_in = ":requirements_3_12.in",
    requirements_txt = ":requirements_3_12.txt",
    python_version = "3_12",
)

py_binary(
  name = "hello_world",
  srcs = ["hello_world.py"],
  deps = [
    requirement("numpy"),
  ]
)
