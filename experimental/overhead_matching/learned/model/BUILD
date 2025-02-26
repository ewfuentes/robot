
load("@pip//:requirements.bzl", "requirement")

py_library(
  name="pose_optimizer",
  srcs=["pose_optimizer.py"],
  deps=[
    "//common:torch",
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
    "//common:torch"
  ]
)

py_test(
  name="clevr_tokenizer_test",
  srcs=["clevr_tokenizer_test.py"],
  data=["@clevr_test_set//:files"],
  deps=[
    requirement("numpy"),
    "//experimental/overhead_matching/learned/data:clevr_dataset",
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
    "//common:torch",
  ]
)

py_test(
  name="clevr_transformer_test",
  srcs=["clevr_transformer_test.py"],
  data=["@clevr_test_set//:files"],
  deps=[
    "//experimental/overhead_matching/learned/data:clevr_dataset",
    ":clevr_transformer",
    ":clevr_tokenizer",
  ]
)
