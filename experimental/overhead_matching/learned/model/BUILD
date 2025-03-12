
load("@pip//:requirements.bzl", "requirement")

py_library(
  name="pose_optimizer",
  srcs=["pose_optimizer.py"],
  deps=[
    "//common/torch:load_torch_deps",
    requirement("sdprlayers"),
    requirement("numpy"),
    requirement("sympy"),
  ]
)

py_test(
  name="pose_optimizer_test",
  srcs=["pose_optimizer_test.py"],
  deps=[
    ":pose_optimizer",
  ]
)

py_library(
  name="clevr_tokenizer",
  srcs=["clevr_tokenizer.py"],
  visibility=[
    "//experimental/overhead_matching/learned:__subpackages__",
    "//common/python:__subpackages__",
  ],
  deps=[
    ":perspective_projection",
    "//experimental/overhead_matching/learned/data:clevr_dataset",
    "//common/torch:load_torch_deps"
  ]
)

py_test(
  name="clevr_tokenizer_test",
  srcs=["clevr_tokenizer_test.py"],
  data=["@clevr_test_set//:files"],
  deps=[
    requirement("numpy"),
    ":clevr_tokenizer",
  ]
)

py_library(
  name="clevr_transformer",
  srcs=["clevr_transformer.py"],
  visibility=[
    "//experimental/overhead_matching/learned:__subpackages__",
    "//common/python:__subpackages__",
  ],
  deps=[
    "//common/torch:load_torch_deps",
    ":pose_optimizer",
    ":clevr_tokenizer",
  ]
)

py_test(
  name="clevr_transformer_test",
  srcs=["clevr_transformer_test.py"],
  data=["@clevr_test_set//:files"],
  deps=[
    "//experimental/overhead_matching/learned/data:clevr_dataset",
    ":clevr_transformer",
  ]
)

py_library(
  name="perspective_projection",
  srcs=["perspective_projection.py"],
  deps=[
    "//common/torch:load_torch_deps"
  ]
)


py_test(
  name="perspective_projection_test",
  srcs=["perspective_projection_test.py"],
  deps=[
    ":perspective_projection",
  ]
)
