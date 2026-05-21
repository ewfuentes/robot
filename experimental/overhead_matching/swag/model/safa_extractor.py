
import common.torch.load_torch_deps
import torch

from pathlib import Path

from common.torch.load_and_save_models import load_model
from experimental.overhead_matching.swag.model.swag_config_types import SafaExtractorConfig, ExtractorDataRequirement
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.patch_embedding import WagPatchEmbedding


class SafaExtractor(torch.nn.Module):
    def __init__(self, config: SafaExtractorConfig):
        super().__init__()
        model = load_model(Path(config.model_path),
                           device="cpu",
                           skip_consistent_output_check=True)
        # Unwrap torch.compile'd OptimizedModule to get the original module.
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        self._inner_model = model
        self._is_wag = isinstance(self._inner_model, WagPatchEmbedding)

        if config.freeze:
            for param in self._inner_model.parameters():
                param.requires_grad = False
            self._inner_model.eval()

        self._freeze = config.freeze

    def train(self, mode=True):
        super().train(mode)
        if self._freeze:
            self._inner_model.eval()
        return self

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        if self._is_wag:
            embeddings, _ = self._inner_model(model_input.image)
        else:
            embeddings, _ = self._inner_model(model_input)

        batch_size, num_embeddings, _ = embeddings.shape
        positions = torch.zeros(batch_size, num_embeddings, 1, 2,
                                device=embeddings.device)
        mask = torch.zeros(batch_size, num_embeddings,
                           dtype=torch.bool, device=embeddings.device)

        return ExtractorOutput(
            features=embeddings,
            positions=positions,
            mask=mask)

    @property
    def output_dim(self):
        return self._inner_model.output_dim

    @property
    def num_position_outputs(self):
        return 1

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return [ExtractorDataRequirement.IMAGES]
