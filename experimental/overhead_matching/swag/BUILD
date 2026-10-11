load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

# The one wheel loci_ros installs: everything from here that runs on the rig. Add a
# target to the py_package's deps to ship it. Only first-party code goes in -- torch,
# torchvision, msgspec, numpy and tqdm are declared as requirements and come from the
# installing environment, where torch has to match the GPU and CUDA anyway -- so it is
# pure Python, py3-none-any.
py_package(
  name = "swag_package",
  packages = ["common", "experimental"],
  deps = [
    "//experimental/overhead_matching/swag/data:landmark_correspondence_dataset",
    "//experimental/overhead_matching/swag/data:vigor_dataset",
    "//experimental/overhead_matching/swag/evaluation:correspondence_matching",
    "//experimental/overhead_matching/swag/filter:histogram_belief",
    "//experimental/overhead_matching/swag/filter:particle_filter",
    "//experimental/overhead_matching/swag/model:landmark_correspondence_model",
    "//experimental/overhead_matching/swag/model:patch_embedding",
    "//experimental/overhead_matching/swag/model:semantic_landmark_extractor",
    "//experimental/overhead_matching/swag/scripts:correspondence_configs",
    "//experimental/overhead_matching/swag/scripts:panorama_to_pinhole_lib",
  ],
)

py_wheel(
  name = "swag_wheel",
  distribution = "overhead_matching_swag",
  version = "0.0.1",
  python_tag = "py3",
  abi = "none",
  platform = "any",
  tags = ["manual"],
  requires = [
    "torch", "torchvision", "msgspec", "numpy", "tqdm",
    # The landmark modules: correspondence matching and its model's config, the
    # extractor's prompt and schema, and vigor_dataset's OSM landmark loading.
    "pandas", "scipy", "psutil", "pydantic", "openai", "ollama", "requests",
    "geopandas", "shapely", "pyarrow", "lmdb", "pyyaml",
    # panorama_to_pinhole: the pinhole crops sent to the landmark extractor.
    "pillow",
  ],
  python_requires = ">=3.12",
  deps = [":swag_package"],
)
