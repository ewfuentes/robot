
import msgspec
from typing import Union

STRUCT_OPTS = {
    "tag": True,
    "tag_field": "kind",
    "frozen": True
}


class DinoFeatureMapExtractorConfig(msgspec.Struct, **STRUCT_OPTS):
    model_str: str = "dinov2_vitb14"


class AlphaEarthExtractorConfig(msgspec.Struct, **STRUCT_OPTS):
    auxiliary_info_key: str
    version: str
    patch_size: tuple[int, int]


class SemanticNullExtractorConfig(msgspec.Struct, **STRUCT_OPTS):
    ...


class SemanticEmbeddingMatrixConfig(msgspec.Struct, **STRUCT_OPTS):
    vocabulary: list[str]
    embedding_dim: int


class SemanticSegmentExtractorConfig(msgspec.Struct, **STRUCT_OPTS):
    points_per_batch: int = 128
    sam_model_str: str = "facebook/sam2.1-hiera-large"
    clip_model_str: str = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


class SemanticLandmarkExtractorConfig(msgspec.Struct, **STRUCT_OPTS):
    sentence_model_str: str = 'sentence-transformers/all-mpnet-base-v2'


class PlanarPositionEmbeddingConfig(msgspec.Struct, **STRUCT_OPTS):
    min_scale: float
    scale_step: float
    embedding_dim: int


class SphericalPositionEmbeddingConfig(msgspec.Struct, **STRUCT_OPTS):
    scale_step: float
    embedding_dim: int


class TransformerAggregatorConfig(msgspec.Struct, **STRUCT_OPTS):
    num_transformer_layers: int
    num_attention_heads: int
    hidden_dim: int
    dropout_frac: float


FeatureMapExtractorConfig = Union[DinoFeatureMapExtractorConfig, None]
SemanticTokenExtractorConfig = Union[
        SemanticNullExtractorConfig, SemanticEmbeddingMatrixConfig, SemanticSegmentExtractorConfig]
PositionEmbeddingConfig = Union[PlanarPositionEmbeddingConfig, SphericalPositionEmbeddingConfig]
AggregationConfig = Union[TransformerAggregatorConfig]

ExtractorConfig = Union[
    DinoFeatureMapExtractorConfig,
    AlphaEarthExtractorConfig,
    SemanticNullExtractorConfig,
    SemanticEmbeddingMatrixConfig,
    SemanticSegmentExtractorConfig,
    SemanticLandmarkExtractorConfig,
]
