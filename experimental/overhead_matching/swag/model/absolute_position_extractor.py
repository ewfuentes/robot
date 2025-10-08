
import common.torch.load_torch_deps
import torch
import math
import numpy as np

from experimental.overhead_matching.swag.model.swag_config_types import (
    AbsolutePositionExtractorConfig, ExtractorDataRequirement)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)


class AbsolutePositionExtractor(torch.nn.Module):
    def __init__(self, config: AbsolutePositionExtractorConfig):
        super().__init__()
        self.embedding_dim = 256
        self._scale_step = 2

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        batch_size = len(model_input.metadata)

        mask = torch.zeros((batch_size, 1), dtype=torch.bool)
        features = torch.zeros((batch_size, 1, self.output_dim))
        positions = torch.zeros((batch_size, 1, self.num_position_outputs, 2))

        lat_lon_tensor = torch.zeros((batch_size, 2), device=model_input.image.device)

        for batch_item in range(batch_size):
            lat, lon = model_input.metadata[batch_item]['lat'],  model_input.metadata[batch_item]['lon']
            lat_lon_tensor[batch_item, 0] = lat
            lat_lon_tensor[batch_item, 1] = lon

        lat_lon_tensor = torch.deg2rad(lat_lon_tensor)
        num_scales = self.embedding_dim // 4
        for scale_idx in range(num_scales):
            embedding_idx_start = 4 * scale_idx
            scale = (self._scale_step ** scale_idx)
            features[..., 0, embedding_idx_start + 0] = torch.sin(lat_lon_tensor[:, 0] * scale)
            features[..., 0, embedding_idx_start + 1] = torch.cos(lat_lon_tensor[:, 0] * scale)
            features[..., 0, embedding_idx_start + 2] = torch.sin(lat_lon_tensor[:, 1] * scale)
            features[..., 0, embedding_idx_start + 3] = torch.cos(lat_lon_tensor[:, 1] * scale)

        return ExtractorOutput(
            features=features.to(model_input.image.device),
            mask=mask.to(model_input.image.device),
            positions=positions.to(model_input.image.device))

    @property
    def output_dim(self):
        return self.embedding_dim

    @property
    def num_position_outputs(self):
        return 1

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return []
