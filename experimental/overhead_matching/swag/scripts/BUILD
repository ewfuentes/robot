
load("@pip//:requirements.bzl", "requirement")

py_binary(
  name="train",
  srcs=["train.py"],
  deps=[
    requirement("torch"),
    requirement("tqdm"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/model:patch_embedding",
  ]
)
