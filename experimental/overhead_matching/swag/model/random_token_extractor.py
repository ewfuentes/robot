import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.model.swag_config_types import (
    ExtractorDataRequirement, RandomTokenExtractorConfig,
)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput,
)


class RandomTokenExtractor(torch.nn.Module):
    """Emits a fixed number of standard-normal noise tokens, resampled per forward.

    Used for Stage 1 of the SAFA-distillation pretraining: gives the aggregator a
    landmark-shaped slot of distractor tokens to learn to ignore, so the
    architecture matches Stage 2 (where these slots will be filled with real
    landmark tokens).
    """

    def __init__(self, config: RandomTokenExtractorConfig):
        super().__init__()
        self._num_tokens = config.num_tokens
        self._raw_dim = config.raw_dim

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        dev = model_input.image.device
        batch_size = model_input.image.shape[0]
        features = torch.randn(
            batch_size, self._num_tokens, self._raw_dim, device=dev)
        positions = torch.zeros(
            batch_size, self._num_tokens, 1, 2, device=dev)
        mask = torch.zeros(
            batch_size, self._num_tokens, dtype=torch.bool, device=dev)
        return ExtractorOutput(features=features, positions=positions, mask=mask)

    @property
    def output_dim(self):
        return self._raw_dim

    @property
    def num_position_outputs(self):
        return 1

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return [ExtractorDataRequirement.IMAGES]
