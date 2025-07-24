
import common.torch.load_torch_deps
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInput:
    image: torch.Tensor
    metadata: list[dict]
    cached_tensors: dict[str, Any] = field(default_factory=dict)

    def to(self, *args, **kwargs):
        return ModelInput(
            image=self.image.to(*args, **kwargs),
            metadata=self.metadata,
            cached_tensors=(
                None if self.cached_tensors is None
                else {k: v.to(*args, **kwargs) for k, v in self.cached_tensors.items()}))


@dataclass
class FeatureMapExtractorOutput:
    features: torch.Tensor

    @classmethod
    def collate(cls, items: list["FeatureMapExtractorOutput"]):
        return FeatureMapExtractorOutput(
            features=torch.stack([x.features for x in items], dim=0))

    def to(self, *args, **kwargs):
        return FeatureMapExtractorOutput(features=self.features.to(*args, **kwargs))


@dataclass
class SemanticTokenExtractorOutput:
    features: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor

    @classmethod
    def collate(cls, items: list["SemanticTokenExtractorOutput"]):
        return SemanticTokenExtractorOutput(
            features=pad_sequence([x.features for x in items], batch_first=True),
            positions=pad_sequence([x.positions for x in items], batch_first=True),
            mask=pad_sequence([x.mask for x in items], batch_first=True, padding_value=True))

    def to(self, *args, **kwargs):
        return SemanticTokenExtractorOutput(
            features=self.features.to(*args, **kwargs),
            positions=self.positions.to(*args, **kwargs),
            mask=self.mask.to(*args, **kwargs))
