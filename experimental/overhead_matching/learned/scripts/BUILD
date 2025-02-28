
load("@pip//:requirements.bzl", "requirement")

py_binary(
  name="train_clevr_transformer",
  srcs=["train_clevr_transformer.py"],
  deps=[
    requirement("numpy"),
    requirement("torchvision"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/learned/data:clevr_dataset",
    "//experimental/overhead_matching/learned/model:clevr_tokenizer",
    "//experimental/overhead_matching/learned/model:clevr_transformer",
  ]
)

py_binary(
  name="evaluate_clevr_transformer",
  srcs=["evaluate_clevr_transformer.py"],
  deps=[
    ":train_clevr_transformer",
    requirement("pandas"),
    requirement("numpy"),
    requirement("matplotlib"),
    requirement("seaborn"),
    requirement("PyGObject"),
    "//common/torch:load_torch_deps",
    "//experimental/overhead_matching/learned/data:clevr_dataset",
    "//experimental/overhead_matching/learned/model:clevr_tokenizer",
    "//experimental/overhead_matching/learned/model:clevr_transformer",
  ]
)

py_test(
  name="sdprlayers_test",
  srcs=["sdprlayers_test.py"],
  deps=[
    requirement("sdprlayers"),
    requirement("asrl-pylgmath"),
    requirement("theseus-ai"),
    requirement("scipy"),
    "//common/torch:load_torch_deps",
  ]
)
