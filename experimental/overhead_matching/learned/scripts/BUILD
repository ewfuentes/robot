
load("@pip//:requirements.bzl", "requirement")

py_binary(
  name="train_clevr_transformer",
  srcs=["train_clevr_transformer.py"],
  deps=[
    requirement("numpy"),
    requirement("IPython"),
    "//common:torch",
    "//experimental/overhead_matching/learned/data:clevr_dataset",
    "//experimental/overhead_matching/learned/model:clevr_tokenizer",
    "//experimental/overhead_matching/learned/model:clevr_transformer",
  ]
)
