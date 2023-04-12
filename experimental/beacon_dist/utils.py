
import torch
import numpy as np
from typing import NamedTuple
from collections import defaultdict

KeypointDescriptorDtype = np.dtype([
    ('image_id', np.int64),
    ('angle', np.float32),
    ('class_id', np.int32),
    ('octave', np.int32),
    ('x', np.float32),
    ('y', np.float32),
    ('response', np.float32),
    ('size', np.float32),
    ('descriptor', np.uint8, (32,))
])

class ReconstructorBatch(NamedTuple):
    # dimension [Batch]
    image_id: torch.Tensor

    # ragged tensors with dimension [Batch, key_point_number]
    angle: torch.Tensor
    class_id: torch.Tensor
    octave: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    response: torch.Tensor
    size: torch.Tensor
    # ragged tensor with dimension [Batch, key_point_number, 32]
    descriptor: torch.Tensor

class Dataset(torch.utils.data.Dataset):
    def __init__(self, *, data: None | np.ndarray = None, filename: None | str = None):
        if filename:
            data = np.load(filename)
        assert data is not None

        tensors = defaultdict(list)
        _, first_idx = np.unique(data['image_id'], return_index=True)
        for subset in np.split(data, first_idx[1:]):
            for field in data.dtype.names:
                if field == 'image_id':
                    tensors[field].append(torch.from_numpy(np.array(subset[0]['image_id'])))
                    continue
                tensors[field].append(torch.from_numpy(subset[field]))

        self._data = ReconstructorBatch(**{key: torch.nested.nested_tensor(value) for key, value in tensors.items()})

    def __getitem__(self, idx: int):
        return ReconstructorBatch(**{f: getattr(self._data, f)[idx] for f in self._data._fields})

    def __len__(self):
        return self._data.image_id.size(0)

    def __repr__(self):
        return f'<Dataset: {len(self)} examples>'
