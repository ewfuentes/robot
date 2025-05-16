import json
import copy
from pathlib import Path
from datetime import datetime
import common.torch.load_torch_deps
import torch
from toolchain import git_info
import base64

from dataclasses import is_dataclass
from collections import namedtuple
from typing import Any, Union
import dataclasses
import numpy as np


def get_git_commit_hash():
    """Returns None if fails to get commit hash (e.g. not in a git repo)"""
    return git_info.STABLE_GIT_COMMIT


def get_git_diff():
    """Returns None if fails to get diff (either not in repo or no active changes)"""
    if hasattr(git_info, "STABLE_GIT_DIFF"):
        return base64.b64decode(git_info.STABLE_GIT_DIFF).decode('utf-8')
    else:
        return None


def deep_equal(a, b, rtol=1e-5, atol=1e-8, print_reason: bool = False):
    """
    Compare two values for equality, handling nested structures and tensor types.

    Args:
        a: First value
        b: Second value
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison

    Returns:
        bool: True if values are equal within tolerance
    """

    # Handle None values
    if a is None and b is None:
        return True
    if a is None or b is None:
        if print_reason:
            print("A or B is None but not both")
        return False
    # Handle different types
    if type(a) != type(b):
        if print_reason:
            print("Types do not match", type(a), type(b))
        return False

    # Handle tensors
    if isinstance(a, torch.Tensor):
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            if print_reason:
                print("Not all values are close torch", a[~torch.isclose(
                    a, b, rtol, atol)] - b[~torch.isclose(a, b, rtol, atol)])
            return False
        return True

    # Handle numpy arrays
    if isinstance(a, np.ndarray):
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            if print_reason:
                print("Not all values are close numpy", a[~np.isclose(
                    a, b, rtol, atol)], b[~np.isclose(a, b, rtol, atol)])
            return False
        return True

    # Handle dictionaries
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            if print_reason:
                print("Keys do not match in a dict")
            return False
        out = all(deep_equal(a[k], b[k], rtol=rtol, atol=atol) for k in a)
        if not out and print_reason:
            print("Values don't match for a key in a dict", [
                  k for k in a if not deep_equal(a[k], b[k], rtol=rtol, atol=atol, print_reason=True)])
        return out

    # Handle lists and tuples
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            if print_reason:
                print("a and b are list/tuple and have different length")
            return False
        out = all(deep_equal(x, y, rtol=rtol, atol=atol, print_reason=print_reason)
                  for x, y in zip(a, b))
        if not out and print_reason:
            print("All members of list/tuple did not match")
        return out

    # Handle dataclasses
    if hasattr(a, '__dataclass_fields__'):
        out = all(
            deep_equal(getattr(a, f.name), getattr(b, f.name),
                       rtol=rtol, atol=atol, print_reason=print_reason)
            for f in dataclasses.fields(a)
        )
        if not out and print_reason:
            print("All values of dataclass did not match")
        return out

    # Default comparison for other types
    out = a == b
    if not out and print_reason:
        print("a didn't equal b. Plain and simple")
    return out


def to_device(obj: Any, device: Union[str, torch.device]) -> Any:
    """
    Recursively moves all tensors in an object to the specified device.

    Args:
        obj: Object containing tensors (can be a tensor, list, tuple, dict, dataclass, or namedtuple)
        device: Target device ('cpu', 'cuda', or torch.device object)

    Returns:
        Same object with all tensors moved to specified device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    elif isinstance(obj, (list, tuple)):
        # Handle lists and tuples
        converted = [to_device(item, device) for item in obj]

        if isinstance(obj, tuple):
            # Handle named tuples
            if hasattr(obj, '_fields'):  # Check if it's a namedtuple
                return type(obj)(*converted)
            return tuple(converted)
        return converted

    elif isinstance(obj, dict):
        # Handle dictionaries
        return {key: to_device(value, device) for key, value in obj.items()}

    elif is_dataclass(obj):
        # Handle dataclasses
        return type(obj)(**{
            field.name: to_device(getattr(obj, field.name), device)
            for field in obj.__dataclass_fields__.values()
        })

    # Return unchanged if not a known type
    return obj


def save_model(
    model: torch.nn.Module,
    save_path: Path,  # folder to save model in
    example_model_inputs: tuple,  # inputs to forward call
    aux_information: dict | None = None,  # auxilary information to save (serializable dict)
) -> None:

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    # make output directory
    save_path.mkdir(exist_ok=False, parents=True)

    # serialize model
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    with torch.no_grad():
        model_out = model_copy(*example_model_inputs)

    # save weights alone
    torch.save(model_copy.state_dict(), save_path / "model_weights.pt")
    # save entire model
    torch.save(model_copy, save_path / "model.pt")

    # dump aux information
    if aux_information is None:
        aux_information = {}
    assert 'current_time' not in aux_information
    aux_information['current_time'] = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    with (save_path / "aux.json").open("w") as f:
        json.dump(aux_information, f)

    # dump commit information
    with (save_path / "commit_hash.txt").open("w") as f:
        commit_hash = get_git_commit_hash()
        if commit_hash is None:
            print("WARNING: Could not get git commit hash when saving model")
        f.write(commit_hash if commit_hash is not None else "Could not get git commit hash")
    with (save_path / "diff.txt").open("w") as f:
        git_diff = get_git_diff()
        if git_diff is None:
            print("WARNING: Could not get git diff when saving model")
        f.write(git_diff if git_diff is not None else "Could not get git diff")

    # dump expected behavior:
    torch.save({"input": example_model_inputs, "output": model_out}, save_path / "input_output.tar")


def load_model(
    load_path: Path,  # folder where model is saved
    device: str = "cpu",
    *,
    skip_constient_output_check: bool = False
):

    if not isinstance(load_path, Path):
        load_path = Path(load_path)

    # load model
    model = torch.load(load_path / "model.pt", map_location=device, weights_only=False)
    model.eval()

    # verify model
    if not skip_constient_output_check:
        input_output = torch.load(load_path / 'input_output.tar',
                                  map_location=device, weights_only=False)
        with torch.no_grad():
            new_output = model(*input_output['input'])
        # observed 1e-6 differences when comparing cpu tensors to gpu tensors
        assert deep_equal(new_output, input_output['output'], atol=1e-3, print_reason=False)
    return model
