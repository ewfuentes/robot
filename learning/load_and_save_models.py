import json
import copy
from pathlib import Path
from datetime import datetime
import subprocess
import common.torch as torch
from dataclasses import is_dataclass
from collections import namedtuple
from typing import Any, Union
import dataclasses
import numpy as np


def get_git_commit_hash():
    try:
        current_directory = subprocess.check_output(
            ["pwd"],
            stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
        print('current_directory:', current_directory)
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
        print('repo root:', repo_root)
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        commit_hash = None
        print("Error retrieving commit hash:", e.output.decode("utf-8"))
    return commit_hash


def get_git_diff():
    try:
        diff = subprocess.check_output(
            ["git", "diff"],
            stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        diff = None
        print("Error retrieving git diff:", e.output.decode("utf-8"))
    return diff


def deep_equal(a, b, rtol=1e-5, atol=1e-8):
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
        return False

    # Handle different types
    if type(a) != type(b):
        return False

    # Handle tensors
    if isinstance(a, torch.Tensor):
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            return False
        return True

    # Handle numpy arrays
    if isinstance(a, np.ndarray):
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            return False
        return True

    # Handle dictionaries
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal(a[k], b[k], rtol=rtol, atol=atol) for k in a)

    # Handle lists and tuples
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b))

    # Handle dataclasses
    if hasattr(a, '__dataclass_fields__'):
        return all(
            deep_equal(getattr(a, f.name), getattr(b, f.name), rtol=rtol, atol=atol)
            for f in dataclasses.fields(a)
        )

    # Default comparison for other types
    return a == b


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
        f.write(get_git_commit_hash())
    with (save_path / "diff.txt").open("w") as f:
        f.write(get_git_diff())

    # dump expected behavior:
    torch.save({"input": example_model_inputs, "output": model_out}, save_path / "input_output.tar")


def load_model(
    load_path: Path,  # folder where model is saved
    device: str = "cpu"
):

    if not isinstance(load_path, Path):
        load_path = Path(load_path)

    # load model
    model = torch.load(load_path / "model.pt", map_location=device, weights_only=False)
    model.eval()

    # verify model
    input_output = torch.load(load_path / 'input_output.tar', map_location=device, weights_only=False)
    new_output = model(*input_output['input'])
    assert deep_equal(new_output, input_output['output'])
    return model
