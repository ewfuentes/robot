
load("@pip//:requirements.bzl", "requirement")

py_binary(
  name="train_clevr_transformer",
  srcs=["train_clevr_transformer.py"],
  visibility=["//common/python:__subpackages__"],
  deps=[
    requirement("numpy"),
    requirement("torchvision"),
    requirement("tensorboard"),
    "//common/torch:load_torch_deps",
    "//common/torch:load_and_save_models",
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
    "//common/torch:load_and_save_models",
    "//experimental/overhead_matching/learned/data:clevr_dataset",
    "//experimental/overhead_matching/learned/model:clevr_tokenizer",
    "//experimental/overhead_matching/learned/model:clevr_transformer",
  ]
)

py_binary(
  name="tmp_diff",
  srcs=["tmp_diff.py"],
  deps=[
    "//toolchain:git_info",
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
