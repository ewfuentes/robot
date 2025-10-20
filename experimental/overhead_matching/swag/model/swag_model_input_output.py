
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


@dataclass
class ExtractorOutput:
    features: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor
    debug: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def collate(cls, items: list["SemanticTokenExtractorOutput"]):
        return ExtractorOutput(
            features=pad_sequence([x.features for x in items], batch_first=True),
            positions=pad_sequence([x.positions for x in items], batch_first=True),
            mask=pad_sequence([x.mask for x in items], batch_first=True, padding_value=True),
            debug={
                k: pad_sequence([x.debug[k] for x in items], batch_first=True)
                for k in items[0].debug})

    def to(self, *args, **kwargs):
        return ExtractorOutput(
            features=self.features.to(*args, **kwargs),
            positions=self.positions.to(*args, **kwargs),
            mask=self.mask.to(*args, **kwargs),
            debug={k: v.to(*args, **kwargs) for k, v in self.debug.items()})


def derive_data_requirements_from_model(model, use_cached_extractors=None):
    """
    Derive what data (images, landmarks) a model requires from its extractors.

    Args:
        model: SwagPatchEmbedding or WagPatchEmbedding instance
        use_cached_extractors: list of extractor names that use caches (ignored for requirements)

    Returns:
        set of ExtractorDataRequirement values
    """
    from experimental.overhead_matching.swag.model.swag_config_types import ExtractorDataRequirement

    if use_cached_extractors is None:
        use_cached_extractors = []

    requirements = set()

    # Handle SwagPatchEmbedding - has _extractor_by_name
    if hasattr(model, '_extractor_by_name'):
        for name, extractor in model._extractor_by_name.items():
            # Skip cached extractors - they don't need raw data
            if name in use_cached_extractors:
                continue
            if hasattr(extractor, 'data_requirements'):
                requirements.update(extractor.data_requirements)

    # Handle WagPatchEmbedding - check its data_requirements property
    elif hasattr(model, 'data_requirements'):
        requirements.update(model.data_requirements)

    return requirements
