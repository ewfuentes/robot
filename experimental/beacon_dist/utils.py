import torch
import numpy as np
from typing import NamedTuple, Callable
from collections import defaultdict

KeypointDescriptorDtype = np.dtype(
    [
        ("image_id", np.int64),
        ("angle", np.float32),
        ("class_id", np.int32),
        ("octave", np.int32),
        ("x", np.float32),
        ("y", np.float32),
        ("response", np.float32),
        ("size", np.float32),
        ("descriptor", np.int16, (32,)),
        ("class_label", np.int64),
    ]
)


class KeypointBatch(NamedTuple):
    # dimension [Batch]
    image_id: torch.Tensor

    # Note that the following tensors maybe ragged
    # Tensors with dimension [Batch, key_point_number]
    angle: torch.Tensor
    class_id: torch.Tensor
    octave: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    response: torch.Tensor
    size: torch.Tensor
    class_label: torch.Tensor

    # Tensor with dimension [Batch, key_point_number, 32]
    descriptor: torch.Tensor

    def to(self, *args, **kwargs):
        return KeypointBatch(
            **{key: value.to(*args, **kwargs) for key, value in self._asdict().items()}
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        data: None | np.ndarray = None,
        filename: None | str = None,
        sample_transform_fn: None | Callable[[KeypointBatch], KeypointBatch] = None,
    ):
        if filename:
            data = np.load(filename)
        assert data is not None

        tensors = defaultdict(list)
        _, first_idx = np.unique(data["image_id"], return_index=True)
        for subset in np.split(data, first_idx[1:]):
            for field in data.dtype.names:
                if field == "image_id":
                    tensors[field].append(
                        torch.from_numpy(np.array(subset[0]["image_id"]))
                    )
                    continue
                tensors[field].append(torch.from_numpy(subset[field].copy()))

        self._data = KeypointBatch(
            **{key: torch.nested.nested_tensor(value) for key, value in tensors.items()}
        )
        self._sample_transform_fn = (
            sample_transform_fn if sample_transform_fn is not None else lambda x: x
        )

    def __getitem__(self, idx: int) -> KeypointBatch:
        # unsqueeze to add a batch dimension
        return self._sample_transform_fn(
            KeypointBatch(
                **{f: getattr(self._data, f)[idx] for f in self._data._fields}
            )
        )

    def __len__(self) -> int:
        return self._data.image_id.size(0)

    def __repr__(self) -> str:
        return f"<Dataset: {len(self)} examples>"

    def data(self) -> KeypointBatch:
        return self._data


def sample_keypoints(
    sample: KeypointBatch, num_keypoints_to_sample: int, gen: torch.Generator
) -> KeypointBatch:
    # This function only works on single sample
    assert sample.image_id.ndim == 0

    num_keypoints = len(sample.x)
    if num_keypoints > num_keypoints_to_sample:
        # We need to sample the data
        idxs = torch.multinomial(
            torch.arange(0, num_keypoints, dtype=torch.float32),
            num_samples=num_keypoints_to_sample,
            replacement=False,
            generator=gen,
        )
        new_fields = {}
        for field_name in sample._fields:
            field = getattr(sample, field_name)
            if field.ndim > 0:
                new_fields[field_name] = field[idxs, ...]
            else:
                new_fields[field_name] = field
        return KeypointBatch(**new_fields)
    return sample


def batchify(sample: KeypointBatch) -> KeypointBatch:
    new_fields = {}
    for field_name in sample._fields:
        field = getattr(sample, field_name)
        if field.ndim == 1:
            new_fields[field_name] = torch.stack(field.unbind())
        else:
            info = torch.finfo if field.dtype.is_floating_point else torch.iinfo
            padding_value = info(field.dtype).min
            new_fields[field_name] = torch.nested.to_padded_tensor(
                field, padding=padding_value
            )

    return KeypointBatch(**new_fields)


def reconstruction_loss(
    input_batch: KeypointBatch,
    model_output: KeypointBatch,
    fields_to_compare: None | list[str] = None,
) -> torch.Tensor:
    if fields_to_compare is None:
        fields_to_compare = ["x", "y"]

    loss = torch.tensor(0.0)
    for field_name in fields_to_compare:
        assert field_name in input_batch._fields
        input_field = getattr(input_batch, field_name)
        output_field = getattr(model_output, field_name)
        loss += torch.nn.functional.mse_loss(input_field, output_field)
    return loss


def is_valid_configuration(class_labels: torch.Tensor, query: torch.Tensor):
    # a configuration is valid if all exclusive keypoints are present
    assert class_labels.shape == query.shape
    exclusive_keypoints = np.remainder(np.log2(class_labels), 1.0) < 1e-6
    unique_classes = torch.unique(exclusive_keypoints * class_labels)

    is_valid_mask = torch.ones((query.shape[0], 1), dtype=torch.bool)
    for class_name in unique_classes:
        if class_name == 0:
            continue
        class_mask = class_labels == class_name
        class_present_in_query = torch.logical_and(class_mask, query)
        class_matches = torch.all(class_mask == class_present_in_query, dim=1, keepdim=True)
        is_valid_mask = torch.logical_and(class_matches, is_valid_mask)

    return is_valid_mask.to(torch.float32)


def valid_configuration_loss(
    class_labels: torch.Tensor,
    query: torch.Tensor,
    model_output: torch.Tensor,
):
    # Compute if this is a valid configuration
    labels = is_valid_configuration(class_labels, query)

    return torch.nn.functional.binary_cross_entropy_with_logits(model_output, labels)

def generate_valid_queries(class_labels: torch.tensor):
    pass
