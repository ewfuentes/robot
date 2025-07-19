
import common.torch.load_torch_deps
import torch
from dataclasses import dataclass


@dataclass
class ModelInput:
    image: torch.Tensor
    metadata: list[dict]

    def to(self, *args, **kwargs):
        return ModelInput(
            image=self.image.to(*args, **kwargs),
            metadata=self.metadata)


@dataclass
class FeatureMapExtractorOutput:
    features: torch.Tensor


@dataclass
class SemanticTokenExtractorOutput:
    features: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor


