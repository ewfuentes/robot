# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

This repository uses **Bazel** as its build system with support for multiple toolchains and Python versions. **DO NOT** run scripts with python directly if they rely on importing modules in this codebase. To access modules in this codebase in python, setup a bazel target with the appropriate dependencies.

### Building and Testing

```bash
# Build a target
bazel build //path/to/target:name

# Run tests
bazel test //path/to/target:test_name

# Build everything
bazel build //...

# Run all tests
bazel test //...

# Build with specific Python version (default is 3.12)
bazel build --config=python-3.12 //path/to/target

# Generate compile_commands.json for IDE support
bazel run //:refresh_compile_commands
```

### Common Build Configurations

- Default Python version: **3.12** (configured in `.bazelrc`)
- Supported Python versions: 3.8, 3.10, 3.12
- Strip is disabled by default (`--strip=never`)
- Uses fast C++ protos (`--define use_fast_cpp_protos=true`)

## Repository Structure

### Core Components

- **`common/`** - Shared utilities and libraries used across the codebase
  - `geometry/` - Geometric primitives and camera models (C++)
  - `liegroups/` - SO(2), SO(3), SE(2), SE(3) implementations with Python bindings
  - `math/` - Mathematical utilities
  - `torch/` - PyTorch utilities including model loading/saving
  - `python/` - Python-specific utilities and serialization
  - `gps/` - GPS and web mercator utilities
  - `tools/lambda_cloud/` - Lambda Labs cloud integration for launching jobs
  - `openstreetmap/` - OpenStreetMap data parsing
  - `proto/` - Protobuf definitions
  - `testing/` - Test utilities
  - `time/` - Time utilities
  - `video/` - Video processing

- **`experimental/`** - Research and experimental code
  - `overhead_matching/swag/` - Main project for satellite-to-ground image matching
    - `data/` - Dataset loaders (VIGOR, etc.) and satellite embedding databases
    - `model/` - Neural network models (patch embeddings, SWAG models)
    - `scripts/` - Training, evaluation, and utility scripts
    - `evaluation/` - Evaluation algorithms and metrics
    - `filter/` - Filtering algorithms
  - Other experimental projects (beacon_dist, oakd_interface, pokerbots, etc.)

- **`domain/`** - Game theory and decision-making domains (poker, blotto, etc.)

- **`planning/`** - Planning algorithms (A*, Dijkstra, PRM, belief space planning)

- **`learning/`** - Learning algorithms (CFR for game theory)

- **`visualization/`** - Visualization tools (OpenGL, OpenCV)

- **`toolchain/`** - Bazel toolchain configurations for GCC and Clang

- **`third_party/`** - External dependencies managed through WORKSPACE

## Python Development

### Adding Python Dependencies

Python dependencies are managed through requirements files in `third_party/python/`:
- `requirements_3_8.txt`
- `requirements_3_10.txt`
- `requirements_3_12.txt`

To update requirements:
```bash
bazel run //third_party/python:requirements_3_12.update
```

See `common/python/README.md` for detailed instructions on adding new Python versions.

### Python Testing

Python tests use pytest-style tests but are defined as `py_test` targets in BUILD files. Tests typically have the suffix `_test.py`.

## Overhead Matching (SWAG) Project

This is the primary active project in the repository, focused on satellite-to-ground image matching.

### Training Models

Training is done via the `train.py` script with YAML config files:

```bash
# Direct bazel invocation
bazel run //experimental/overhead_matching/swag/scripts:train -- \
  --dataset_base /data/overhead_matching/datasets/VIGOR/ \
  --output_base /tmp/training_outputs \
  --train_config /path/to/config.yaml
```

### Dataset Setup

The project uses the VIGOR dataset for cross-view geo-localization. See `experimental/overhead_matching/swag/data/README.md` for data acquisition instructions.

### Key Model Types

- **Patch Embedding Models** - Extract features from image patches
- **SWAG Models** - Satellite-Weighted Attention Graph models
- **Distance Models** - Various distance metrics (Euclidean, Mahalanobis, learned)

### Evaluation

```bash
# Create evaluation paths
bazel run //experimental/overhead_matching/swag/scripts:create_evaluation_paths

# Evaluate model on paths
bazel run //experimental/overhead_matching/swag/scripts:evaluate_model_on_paths

# Interactive filter debugging
bazel run //experimental/overhead_matching/swag/scripts:step_through_filter
```

## C++ Development

### Toolchains

Multiple toolchains are registered:
- `clang_15_toolchain_for_linux`
- `clang_18_toolchain_for_linux`
- `gcc_10_toolchain_for_linux`
- `gcc_11_toolchain_for_linux`
- `gcc_toolchain_for_linux_aarch64` (cross-compilation)

### Key Libraries

- **Eigen** - Linear algebra (header-only from third_party)
- **Sophus** - Lie groups
- **fmt** - String formatting
- **glog/gflags** - Logging and flags
- **OpenCV** - Computer vision
- **GTSAM** - Factor graph optimization
- **protobuf** - Serialization

### Testing

C++ tests use GoogleTest and are defined as `cc_test` targets.

## Lambda Cloud Integration

The repository includes tools for launching training jobs on Lambda Labs GPU instances:

```bash
# Launch training jobs
bazel run //common/tools/lambda_cloud/lambda_launch:launch_training_jobs
```

See `common/tools/lambda_cloud/README.md` for details.

## TensorBoard

To view training logs:

```bash
bazel run //common/torch:run_tensorboard -- --logdir /path/to/logs
```

## Common Workflows

### Running a Single Python Test

```bash
bazel test //path/to/package:test_name
```

### Running All Tests in a Package

```bash
bazel test //path/to/package/...
```


### Cross-Platform Development

The repository supports cross-compilation for ARM64 via the aarch64 toolchain and includes Jetson sysroot for embedded development.

## Important Notes

- The repository uses custom Python serialization via `msgspec` (see `common/python/serialization.py`)
- For any python script that uses torch, `import common.torch.load_torch_deps` must be imported before `import torch` or CUDA libraries will fail to load.
- Models are saved/loaded with git commit info for reproducibility (see `common/torch/load_and_save_models.py`)
- Bazel 7.x is used with some compatibility workarounds in `.bazelrc`
- LMDB is used for caching tensors to speed up repeated training runs