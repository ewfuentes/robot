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
    ]
)


class ReconstructorBatch(NamedTuple):
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

    # Tensor with dimension [Batch, key_point_number, 32]
    descriptor: torch.Tensor


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        data: None | np.ndarray = None,
        filename: None | str = None,
        sample_transform_fn: None
        | Callable[[ReconstructorBatch], ReconstructorBatch] = None,
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
                tensors[field].append(torch.from_numpy(subset[field]))

        self._data = ReconstructorBatch(
            **{key: torch.nested.nested_tensor(value) for key, value in tensors.items()}
        )
        self._sample_transform_fn = (
            sample_transform_fn if sample_transform_fn is not None else lambda x: x
        )

    def __getitem__(self, idx: int) -> ReconstructorBatch:
        # unsqueeze to add a batch dimension
        return self._sample_transform_fn(
            ReconstructorBatch(
                **{f: getattr(self._data, f)[idx].unsqueeze(0) for f in self._data._fields}
            )
        )

    def __len__(self) -> int:
        return self._data.image_id.size(0)

    def __repr__(self) -> str:
        return f"<Dataset: {len(self)} examples>"


def reconstruction_loss(input_batch, model_output) -> torch.Tensor:
    return torch.tensor(0.0, requires_grad=True)


def sample_keypoints(
    sample: ReconstructorBatch, num_keypoints_to_sample: int, gen: torch.Generator
) -> ReconstructorBatch:
    # This function only works on single sample
    assert sample.image_id.ndim == 1
    assert sample.image_id.size(0) == 1

    num_keypoints= sample.x.shape[1]
    if num_keypoints> num_keypoints_to_sample:
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
            if field.ndim > 1:
                new_fields[field_name] = field[:1, idxs, ...]
            else:
                new_fields[field_name] = field
        return ReconstructorBatch(**new_fields)
    return sample

def batchify(sample: ReconstructorBatch) -> ReconstructorBatch:
    new_fields = {}
    for field_name in sample._fields:
        field = getattr(sample, field_name)
        if field.ndim == 1:
            new_fields[field_name] = torch.stack(field.unbind())
        else:
            info = torch.finfo if field.dtype.is_floating_point else torch.iinfo
            padding_value=info(field.dtype).min
            new_fields[field_name] = torch.nested.to_padded_tensor(field, padding=padding_value)

    return ReconstructorBatch(**new_fields)
