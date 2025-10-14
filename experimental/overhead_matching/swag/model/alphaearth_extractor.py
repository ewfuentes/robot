
import common.torch.load_torch_deps
import torch

from pathlib import Path

import experimental.overhead_matching.swag.data.alphaearth_registry as ar
from experimental.overhead_matching.swag.model.swag_config_types import AlphaEarthExtractorConfig, ExtractorDataRequirement
from experimental.overhead_matching.swag.model.swag_model_input_output import (
        ModelInput, ExtractorOutput)


class AlphaEarthExtractor(torch.nn.Module):
    def __init__(self, config: AlphaEarthExtractorConfig, registry_base_path: Path):
        super().__init__()
        if isinstance(registry_base_path, str):
            registry_base_path = Path(registry_base_path)
        assert registry_base_path.exists()
        self._config = config
        self._registry = ar.AlphaEarthRegistry(registry_base_path, config.version)
        self._patch_size = config.patch_size

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        features_out = []
        position_out = []
        for info in model_input.metadata:
            features, position_info = self._registry.query(lat_deg=info["lat"],
                                                           lon_deg=info["lon"],
                                                           zoom_level=info["zoom_level"],
                                                           patch_size=self._patch_size)
            features = torch.from_numpy(features)
            position_info = torch.from_numpy(position_info)
            features = features.reshape(-1, features.shape[-1])
            position_info = position_info.reshape(-1, 2)
            position_info -= torch.tensor([[info["web_mercator_y"], info["web_mercator_x"]]])

            features_out.append(features)
            position_out.append(position_info)
        features_out = torch.stack(features_out, dim=0).to(torch.float32)
        position_out = torch.stack(position_out, dim=0).to(torch.float32)
        position_out = position_out[:, :, None, :]
        mask = torch.isnan(features_out[..., 0])
        features_out[mask] = 0.0

        return ExtractorOutput(
            features=features_out.to(model_input.image.device),
            positions=position_out.to(model_input.image.device),
            mask=mask.to(model_input.image.device))

    @property
    def output_dim(self):
        return 64

    @property
    def num_position_outputs(self):
        return 1

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return []
