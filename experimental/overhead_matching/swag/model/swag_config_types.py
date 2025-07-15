
from enum import StrEnum, auto
import msgspec
from typing import Union


class FeatureMapExtractorType(StrEnum):
    DINOV2 = auto()


class DinoFeatureMapExtractorConfig(msgspec.Struct, tag=True, tag_field="kind"):
    model_str: str = "dinov2_vitb14"
    type: FeatureMapExtractorType = FeatureMapExtractorType.DINOV2


class SemanticTokenExtractorType(StrEnum):
    NULL_EXTRACTOR = auto()
    EMBEDDING_MAT = auto()
    SEGMENT_EXTRACTOR = auto()


class SemanticNullExtractorConfig(msgspec.Struct, tag=True, tag_field="kind"):
    type: SemanticTokenExtractorType = SemanticTokenExtractorType.NULL_EXTRACTOR


class SemanticEmbeddingMatrixConfig(msgspec.Struct, tag=True, tag_field="kind"):
    vocabulary: list[str]
    embedding_dim: int
    type: SemanticTokenExtractorType = SemanticTokenExtractorType.EMBEDDING_MAT


class SemanticSegmentExtractorConfig(msgspec.Struct, tag=True, tag_field='kind'):
    sam_model_str: str = "vit_t"
    type: SemanticTokenExtractorType = SemanticTokenExtractorType.SEGMENT_EXTRACTOR


class PositionEmbeddingType(StrEnum):
    PLANAR = auto()
    SPHERICAL = auto()


class PlanarPositionEmbeddingConfig(msgspec.Struct, tag=True, tag_field="kind"):
    min_scale: float
    scale_step: float
    embedding_dim: int
    type: PositionEmbeddingType = PositionEmbeddingType.PLANAR


class SphericalEmbeddingConfig(msgspec.Struct, tag=True, tag_field="kind"):
    type: PositionEmbeddingType = PositionEmbeddingType.SPHERICAL


class AggregationType(StrEnum):
    TRANSFORMER = auto()


class TransformerAggregatorConfig(msgspec.Struct, tag=True, tag_field="kind"):
    num_transformer_layers: int
    num_attention_heads: int
    hidden_dim: int
    dropout_frac: float
    type: AggregationType = AggregationType.TRANSFORMER


FeatureMapExtractorConfig = Union[DinoFeatureMapExtractorConfig]
SemanticTokenExtractorConfig = Union[SemanticNullExtractorConfig, SemanticEmbeddingMatrixConfig]
PositionEmbeddingConfig = Union[PlanarPositionEmbeddingConfig, SphericalEmbeddingConfig]
AggregationConfig = Union[TransformerAggregatorConfig]
