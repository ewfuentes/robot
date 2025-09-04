
import common.torch.load_torch_deps
import torch
import math
import numpy as np

from experimental.overhead_matching.swag.model.swag_config_types import (
        AbsolutePositionExtractorConfig)
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
        positions = torch.zeros((batch_size, 1, 2))

        for batch_item in range(batch_size):
            lat, lon = model_input.metadata[batch_item]['lat'],  model_input.metadata[batch_item]['lon']
            lat = torch.deg2rad(torch.as_tensor(lat))
            lon = torch.deg2rad(torch.as_tensor(lon))

            num_scales = self.embedding_dim // 4
            for scale_idx in range(num_scales):
                embedding_idx_start = 4 * scale_idx
                scale = (self._scale_step ** scale_idx)
                features[batch_item, :, embedding_idx_start + 0] = torch.sin(lat * scale)
                features[batch_item, :, embedding_idx_start + 1] = torch.cos(lat * scale)
                features[batch_item, :, embedding_idx_start + 2] = torch.sin(lon * scale)
                features[batch_item, :, embedding_idx_start + 3] = torch.cos(lon * scale)
            
        return ExtractorOutput(
            features=features.to(model_input.image.device),
            mask=mask.to(model_input.image.device),
            positions=positions.to(model_input.image.device))

    @property
    def output_dim(self):
        return self.embedding_dim
