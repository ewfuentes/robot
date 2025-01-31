
load("@pip//:requirements.bzl", "requirement")


py_library(
  name="clevr_tokenizer",
  srcs=["clevr_tokenizer.py"],
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
