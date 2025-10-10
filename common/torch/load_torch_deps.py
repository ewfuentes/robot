# borrowed from: https://gist.github.com/qxcv/183c2d6cd81f7028b802b232d6a9dd62
"""Hack to load CUDA variables shipped via PyPI. Addresses this Torch bug:
https://github.com/pytorch/pytorch/issues/101314
Copied from PyTorch's __init__.py file, with modifications:
https://github.com/pytorch/pytorch/blob/main/torch/__init__.py
Copyright notice below is from Torch.
"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of the PyTorch source tree.
import ctypes
import sys
import os
import platform
from typing import Dict

def _preload_cuda_deps(lib_folder: str, lib_name: str) -> str:
    """Preloads cuda deps if they could not be found otherwise.

    Returns the directory containing the library.
    """
    # Should only be called on Linux if default path resolution have failed
    assert platform.system() == 'Linux', 'Should only be called on Linux'
    import glob
    lib_path = None
    for path in sys.path:
        nvidia_path = os.path.join(path, 'nvidia')

        if os.path.exists(os.path.join(path, lib_folder)):
            path_to_search = path
        elif os.path.exists(nvidia_path):
            path_to_search = nvidia_path
        else:
            continue
        candidate_lib_paths = glob.glob(os.path.join(path_to_search, lib_folder, 'lib', lib_name))
        lib_contents = glob.glob(os.path.join(nvidia_path, lib_folder, 'lib', '*'))
        if candidate_lib_paths and not lib_path:
            lib_path = candidate_lib_paths[0]
        if lib_path:
            break
    if not lib_path:
        for p in sys.path:
            print(p)
        raise ValueError(f"{lib_name} not found in the system path")
    ctypes.CDLL(lib_path)
    return os.path.dirname(lib_path)


def preload_cuda_deps() -> None:
    cuda_libs: Dict[str, str] = {
        'cublas': 'libcublas.so.*[0-9]',
        'cudnn': 'libcudnn.so.*[0-9]',
        'cuda_nvrtc': 'libnvrtc.so.*[0-9]',
        'cuda_runtime': 'libcudart.so.*[0-9]',
        'cuda_cupti': 'libcupti.so.*[0-9]',
        'cufft': 'libcufft.so.*[0-9]',
        'cufile': 'libcufile.so.*[0-9]',
        'curand': 'libcurand.so.*[0-9]',
        'nvjitlink': 'libnvJitLink.so.*[0-9]',
        'cusparse': 'libcusparse.so.*[0-9]',
        'cusparselt': 'libcusparseLt.so.*[0-9]',
        'cusolver': 'libcusolver.so.*[0-9]',
        'nccl': 'libnccl.so.*[0-9]',
        'nvtx': 'libnvToolsExt.so.*[0-9]',
    }

    # Collect library directories and preload libraries
    lib_dirs = set()
    for lib_folder, lib_name in cuda_libs.items():
        lib_dir = _preload_cuda_deps(lib_folder, lib_name)
        lib_dirs.add(lib_dir)

    # Update LD_LIBRARY_PATH so subprocesses (e.g., torch.compile workers) can find libraries
    if lib_dirs:
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_paths = ':'.join(lib_dirs)
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{new_paths}:{current_ld_path}"
        else:
            os.environ['LD_LIBRARY_PATH'] = new_paths


if platform.processor() != "aarch64":
    preload_cuda_deps()
