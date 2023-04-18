import torch
import numpy as np

from typing import NamedTuple

from experimental.beacon_dist import utils


class ReconstructorParams(NamedTuple):
    descriptor_size: int
    descriptor_embedding_size: int
    position_encoding_factor: float


def expand_descriptor(descriptor_tensor: torch.Tensor):
    BITS_PER_BYTE = 8
    shape = descriptor_tensor.shape
    new_shape = *shape[:-1], shape[-1] * BITS_PER_BYTE
    out = torch.zeros(new_shape)
    for i in range(BITS_PER_BYTE):
        out[:, :, i::BITS_PER_BYTE] = torch.bitwise_and(descriptor_tensor, 1 << i) > 0
    return out


def encode_position(x: torch.Tensor, y: torch.Tensor, output_size: int, factor: float):
    assert output_size % 4 == 0, "Output size must be divisible by 4"
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    out = torch.zeros((*x.shape, output_size), dtype=torch.float32)
    for i in range(0, output_size, 4):
        out[:,:,:, 4 * i + 0] = torch.sin(x / factor ** (4 * i / output_size))
        out[:,:,:, 4 * i + 1] = torch.cos(x / factor ** (4 * i / output_size))
        out[:,:,:, 4 * i + 2] = torch.sin(y / factor ** (4 * i / output_size))
        out[:,:,:, 4 * i + 3] = torch.cos(y / factor ** (4 * i / output_size))
    return out


class Reconstructor(torch.nn.Module):
    def __init__(self, params: ReconstructorParams):
        super().__init__()

        # Ignore token

        # descriptor encoder
        self._descriptor_embedding = torch.nn.Linear(
            params.descriptor_size,
            params.descriptor_embedding_size,
            bias=False,
            dtype=torch.float32,
        )

        # Position encoder

        self._params = params

    def encode_descriptors(self, descriptors: torch.Tensor):
        expanded_descriptors = expand_descriptor(descriptors)
        return self._descriptor_embedding(expanded_descriptors)

    def position_encoding(self, x: torch.Tensor, y: torch.Tensor):
        return encode_position(
            x,
            y,
            self._params.descriptor_embedding_size,
            self._params.position_encoding_factor,
        )

    def forward(self, x: utils.ReconstructorBatch):

        # Encode the descriptors
        embedded_descriptors = self.encode_descriptors(
            x.descriptor
        ) + self.position_encoding(x.x, x.y)

        # Pass through transformer encoder layers

        return x._replace(x=x.x)
