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
    "//experimental/overhead_matching/swag/filter:histogram_belief",
    "//experimental/overhead_matching/swag/filter:particle_filter",
    "//experimental/overhead_matching/swag/model:patch_embedding",
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
  requires = ["torch", "torchvision", "msgspec", "numpy", "tqdm"],
  python_requires = ">=3.12",
  deps = [":swag_package"],
)
