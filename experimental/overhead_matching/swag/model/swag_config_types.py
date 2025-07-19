
from enum import StrEnum, auto
import msgspec
from typing import Union


class FeatureMapExtractorType(StrEnum):
    DINOV2 = auto()


class DinoFeatureMapExtractorConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    model_str: str = "dinov2_vitb14"
    type: FeatureMapExtractorType = FeatureMapExtractorType.DINOV2


class SemanticTokenExtractorType(StrEnum):
    NULL_EXTRACTOR = auto()
    EMBEDDING_MAT = auto()
    SEGMENT_EXTRACTOR = auto()


class SemanticNullExtractorConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    type: SemanticTokenExtractorType = SemanticTokenExtractorType.NULL_EXTRACTOR


class SemanticEmbeddingMatrixConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    vocabulary: list[str]
    embedding_dim: int
    type: SemanticTokenExtractorType = SemanticTokenExtractorType.EMBEDDING_MAT


class SemanticSegmentExtractorConfig(msgspec.Struct, tag=True, tag_field='kind', frozen=True):
    points_per_batch: int = 128
    sam_model_str: str = "facebook/sam2.1-hiera-large"
    clip_model_str: str = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    type: SemanticTokenExtractorType = SemanticTokenExtractorType.SEGMENT_EXTRACTOR


class PositionEmbeddingType(StrEnum):
    PLANAR = auto()
    SPHERICAL = auto()


class PlanarPositionEmbeddingConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    min_scale: float
    scale_step: float
    embedding_dim: int
    type: PositionEmbeddingType = PositionEmbeddingType.PLANAR


class SphericalPositionEmbeddingConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    scale_step: float
    embedding_dim: int
    type: PositionEmbeddingType = PositionEmbeddingType.SPHERICAL


class AggregationType(StrEnum):
    TRANSFORMER = auto()


class TransformerAggregatorConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    num_transformer_layers: int
    num_attention_heads: int
    hidden_dim: int
    dropout_frac: float
    type: AggregationType = AggregationType.TRANSFORMER


FeatureMapExtractorConfig = Union[DinoFeatureMapExtractorConfig]
SemanticTokenExtractorConfig = Union[
        SemanticNullExtractorConfig, SemanticEmbeddingMatrixConfig, SemanticSegmentExtractorConfig]
PositionEmbeddingConfig = Union[PlanarPositionEmbeddingConfig, SphericalPositionEmbeddingConfig]
AggregationConfig = Union[TransformerAggregatorConfig]
