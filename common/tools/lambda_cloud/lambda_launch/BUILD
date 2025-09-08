package(default_visibility=["//visibility:public"], features=["warning_compile_flags"])

load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
  name = "config",
  srcs = ["config.py"],
  deps = [
    requirement("pyyaml"),
  ],
)

py_library(
  name = "remote_executor", 
  srcs = ["remote_executor.py"],
  deps = [
    requirement("paramiko"),
  ],
)

py_library(
  name = "job_manager",
  srcs = ["job_manager.py"],
  deps = [
    ":config",
    ":remote_executor",
  ],
)


py_library(
  name = "launcher",
  srcs = ["launcher.py"],
  deps = [
    ":config",
    ":job_manager", 
    "//common/tools/lambda_cloud/lambda_api:lambda_cloud",
  ],
)

py_binary(
  name = "launch_training_jobs",
  srcs = ["launch_training_jobs.py"],
  deps = [
    ":launcher",
  ],
)

py_binary(
  name = "remote_monitor",
  srcs = ["remote_monitor.py"],
  deps = [
    "//common/tools/lambda_cloud/lambda_api:lambda_cloud",
  ],
)