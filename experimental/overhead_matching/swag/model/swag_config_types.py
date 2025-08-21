
import msgspec
from typing import Union


class DinoFeatureMapExtractorConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    model_str: str = "dinov2_vitb14"


class AlphaEarthExtractorConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    version: str
    patch_size: tuple[int, int]


class SemanticNullExtractorConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    ...


class SemanticEmbeddingMatrixConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    vocabulary: list[str]
    embedding_dim: int


class SemanticSegmentExtractorConfig(msgspec.Struct, tag=True, tag_field='kind', frozen=True):
    points_per_batch: int = 128
    sam_model_str: str = "facebook/sam2.1-hiera-large"
    clip_model_str: str = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


class PlanarPositionEmbeddingConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    min_scale: float
    scale_step: float
    embedding_dim: int


class SphericalPositionEmbeddingConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
    scale_step: float
    embedding_dim: int


class TransformerAggregatorConfig(msgspec.Struct, tag=True, tag_field="kind", frozen=True):
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
]
