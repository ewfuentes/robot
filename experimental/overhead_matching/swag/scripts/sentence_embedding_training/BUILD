
load("@pip//:requirements.bzl", "requirement")

py_binary(
  name="train_sentence_embedder",
  srcs=["train_sentence_embedder.py"],
  visibility=["//experimental/overhead_matching/swag/scripts:__pkg__"],
  deps=[
    requirement("torch"),
    requirement("accelerate"),
    requirement("transformers"),
    requirement("datasets"),
    requirement("sentence-transformers"),
    requirement("tqdm"),
    requirement("tensorboard"),
    "//common/torch:load_torch_deps",
    "//common/torch:load_and_save_models",
    "//experimental/overhead_matching/swag/model:trainable_sentence_embedder",
    "//experimental/overhead_matching/swag/model:swag_config_types",
    "//experimental/overhead_matching/swag/model:semantic_landmark_utils",
    "//experimental/overhead_matching/swag/model:semantic_landmark_extractor",
  ]
)

py_binary(
  name="generate_embeddings_from_model",
  srcs=["generate_embeddings_from_model.py"],
  visibility=["//experimental/overhead_matching/swag/scripts:__pkg__"],
  deps=[
    requirement("torch"),
    requirement("tqdm"),
    requirement("transformers"),
    "//common/torch:load_torch_deps",
    "//common/torch:load_and_save_models",
    "//experimental/overhead_matching/swag/model:semantic_landmark_utils",
    "//experimental/overhead_matching/swag/model:trainable_sentence_embedder",
    "//experimental/overhead_matching/swag/model:swag_config_types",
  ]
)

py_binary(
  name="visualize_dataset",
  srcs=["visualize_dataset.py"],
  visibility=["//experimental/overhead_matching/swag/scripts:__pkg__"],
  deps=[
    requirement("datasets"),
    "//experimental/overhead_matching/swag/model:semantic_landmark_utils",
    "//experimental/overhead_matching/swag/model:semantic_landmark_extractor",
  ]
)

py_test(
  name="test_train_sentence_embedder",
  srcs=["test_train_sentence_embedder.py"],
  deps=[
    ":train_sentence_embedder",
    requirement("torch"),
    requirement("transformers"),
    "//common/torch:load_torch_deps",
    "//common/torch:load_and_save_models",
    "//experimental/overhead_matching/swag/model:trainable_sentence_embedder",
    "//experimental/overhead_matching/swag/model:swag_config_types",
  ]
)
