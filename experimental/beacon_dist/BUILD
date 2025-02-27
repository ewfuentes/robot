
package(features=["warning_compile_flags"])

load("@pip//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@io_bazel_rules_docker//container:container.bzl", "container_push")

py_library(
  name = "multiview_dataset",
  srcs = ["multiview_dataset.py"],
  visibility = ["//common/python:__subpackages__"],
  deps = [
    ":utils",
    "//common/torch",
    requirement("numpy"),
    requirement("tqdm"),
  ]
)

py_test(
  name = "multiview_dataset_test",
  srcs = ["multiview_dataset_test.py"],
  deps = [
    ":multiview_dataset",
    ":utils",
    requirement("numpy"),
  ],
)

py_binary(
  name="generate_letter_dataset",
  srcs=["generate_letter_dataset.py"],
  visibility = ["//common/python:__subpackages__"],
  deps=[
    requirement("opencv-python"),
    requirement("Wand"),
    requirement("numpy"),
    requirement("tqdm"),
    ":utils",
  ]
)

py_binary(
  name = "generate_ycb_dataset",
  srcs = ["generate_ycb_dataset.py"],
  visibility = ["//common/python:__subpackages__"],
  data = [":render_ycb_scene_python.so"],
  deps = [
    requirement("numpy"),
    requirement("tqdm"),
    ":utils",
  ]
)

py_library(
  name = "model",
  srcs = ["model.py"],
  visibility = ["//common/python:__subpackages__"],
  deps = [
    "//common/torch",
  ]
)

py_test(
  name = "model_test",
  srcs = ["model_test.py"],
  deps = [
    ":test_helpers",
    ":model",
    ":multiview_dataset",
  ]
)

py_library(
  name = "utils",
  srcs = ["utils.py"],
  visibility = ["//common/python:__subpackages__"],
  deps = [
    "//common/torch",
    requirement("numpy"),
  ]
)

py_library(
  name = "test_helpers",
  srcs = ["test_helpers.py"],
  deps = [
    requirement("numpy"),
    ":utils",
  ]
)

py_test(
  name = "utils_test",
  srcs = ["utils_test.py"],
  deps = [
    requirement("numpy"),
    ":test_helpers",
    ":multiview_dataset",
    ":utils",
  ]
)

py3_image(
  name = "train",
  base = "//common/docker:cuda_python",
  srcs = ["train.py"],
  visibility = ["//common/python:__subpackages__"],
  tags=["manual"],
  deps = [
    "//common/torch",
    requirement("IPython"),
    requirement("tqdm"),
    ":model",
    ":multiview_dataset",
    ":utils",
  ]
)

container_push(
  name = "train_push",
  format = "Docker",
  image = ":train",
  registry = "docker.io",
  repository = "ewfuentes/robot",
  tag = "train",
  tags = ["manual"],
)

cc_library(
  name = "render_ycb_scene",
  hdrs = ["render_ycb_scene.hh"],
  srcs = ["render_ycb_scene.cc"],
  deps = [
    "//common:argument_wrapper",
    "//common:drake",
    "@opencv//:opencv",
  ]
)

cc_test(
  name = "render_ycb_scene_test",
  srcs = ["render_ycb_scene_test.cc"],
  # Rendering requires running on a machine with an Xserver running, which isn't present
  # on a CI machine.
  tags = ["manual"],
  data = [
      "@drake_models",
  ],
  deps = [
      ":render_ycb_scene",
      "@com_google_googletest//:gtest_main",
  ]
)

pybind_extension(
    name = "render_ycb_scene_python",
    srcs = ["render_ycb_scene_python.cc"],
    deps = [":render_ycb_scene"],
)

