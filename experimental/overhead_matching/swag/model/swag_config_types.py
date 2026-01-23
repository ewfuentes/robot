
import msgspec
from typing import Union, NamedTuple
from enum import StrEnum, auto
from common.python.serialization import MSGSPEC_STRUCT_OPTS


class ExtractorDataRequirement(StrEnum):
    """Specifies what data an extractor needs from the dataset."""
    IMAGES = auto()
    LANDMARKS = auto()


class DinoFeatureMapExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    model_str: str = "dinov2_vitb14"
    use_class_token_only: bool = False


class AlphaEarthExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    auxiliary_info_key: str
    version: str
    patch_size: tuple[int, int]


class SemanticNullExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    ...


class AbsolutePositionExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    ...


class SemanticEmbeddingMatrixConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    vocabulary: list[str]
    embedding_dim: int


class SemanticSegmentExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    points_per_batch: int = 128
    sam_model_str: str = "facebook/sam2.1-hiera-large"
    clip_model_str: str = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


class LandmarkType(StrEnum):
    POINT = "point"
    LINESTRING = "linestring"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"


class SemanticLandmarkExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    landmark_type: LandmarkType
    openai_embedding_size: int  # if smaller than the true embedding dim (1536), will crop and renormalize embedding
    embedding_version: str
    auxiliary_info_key: str


class PanoramaSemanticLandmarkExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    openai_embedding_size: int  # if smaller than the true embedding dim (1536), will crop and renormalize embedding
    embedding_version: str
    auxiliary_info_key: str


class PanoramaProperNounExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for extracting proper noun embeddings from panorama landmarks.

    Proper nouns are business names, street signs, etc. extracted from panorama images.
    Produces one token per proper noun per landmark.
    """
    openai_embedding_size: int
    embedding_version: str
    auxiliary_info_key: str


class PanoramaLocationTypeExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for extracting location type embeddings from panoramas.

    Location type is a scene classification like "urban commercial district".
    Produces exactly one token per panorama.
    """
    openai_embedding_size: int
    embedding_version: str
    auxiliary_info_key: str


class CLIPEmbedMode(StrEnum):
    """Mode for composable CLIP text extractors."""
    PROPER_NOUNS_ONLY = "proper_nouns_only"  # proper nouns via CLIP/hash, sentences via pre-computed
    SENTENCES_ONLY = "sentences_only"
    BOTH = "both"


class TextEncoderType(StrEnum):
    """Type of text encoder to use."""
    CLIP = "clip"
    HASH_BIT = "hash_bit"


class ComposableCLIPPanoExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Composable text encoder for panorama landmarks.

    Supports multiple embedding modes:
    - proper_nouns_only: Encode proper nouns with CLIP/hash, sentences use pre-computed
    - sentences_only: Encode full sentences with CLIP/hash, drop proper nouns/addresses
    - both: Encode both proper nouns and sentences with CLIP/hash
    """
    # CLIP settings (used when encoder_type=clip)
    model_name: str = "openai/clip-vit-large-patch14"
    max_text_length: int = 77
    freeze_encoder: bool = False
    use_gradient_checkpointing: bool = True

    # Data loading
    embedding_version: str = ""
    auxiliary_info_key: str = ""

    # Composability options
    embed_mode: CLIPEmbedMode = CLIPEmbedMode.SENTENCES_ONLY
    encoder_type: TextEncoderType = TextEncoderType.CLIP

    # Hash-bit settings (used when encoder_type=hash_bit)
    hash_bit_dim: int = 256

    # Pre-computed embedding size (for sentence fallback in proper_nouns_only mode)
    precomputed_embedding_size: int = 1536

    # Whether to use a learned projection layer for precomputed embeddings
    # When True (default), projects from precomputed_embedding_size to hash_bit_dim
    # When False, truncates/pads embeddings to match output_dim
    use_sentence_projection: bool = True


class ComposableCLIPOSMExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Composable text encoder for OSM landmarks.

    Supports multiple embedding modes:
    - proper_nouns_only: Encode names/addresses with CLIP/hash, sentences use pre-computed
    - sentences_only: Encode full sentences with CLIP/hash, drop proper nouns/addresses
    - both: Encode both names/addresses and sentences with CLIP/hash

    Extracts name from 'name' tag, address from 'addr:street' and 'addr:housenumber'.
    Supports all geometry types with learned type embeddings (CLIP mode only).
    """
    # CLIP settings (used when encoder_type=clip)
    model_name: str = "openai/clip-vit-large-patch14"
    max_text_length: int = 77
    freeze_encoder: bool = False
    use_gradient_checkpointing: bool = True

    # Data loading
    embedding_version: str = ""
    auxiliary_info_key: str = ""

    # Composability options
    embed_mode: CLIPEmbedMode = CLIPEmbedMode.SENTENCES_ONLY
    encoder_type: TextEncoderType = TextEncoderType.CLIP

    # Hash-bit settings (used when encoder_type=hash_bit)
    hash_bit_dim: int = 256

    # Pre-computed embedding size (for sentence fallback in proper_nouns_only mode)
    precomputed_embedding_size: int = 1536

    # Whether to use a learned projection layer for precomputed embeddings
    # When True (default), projects from precomputed_embedding_size to hash_bit_dim
    # When False, truncates/pads embeddings to match output_dim
    use_sentence_projection: bool = True


class SyntheticLandmarkExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    log_grid_spacing: int
    grid_bounds_px: int
    should_produce_bearing_position_for_pano: bool
    embedding_dim: int


class PlanarPositionEmbeddingConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    min_scale: float
    scale_step: float
    embedding_dim: int


class SphericalPositionEmbeddingConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    scale_step: float
    embedding_dim: int


class NullPositionEmbeddingConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    ...


class TransformerAggregatorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    num_transformer_layers: int
    num_attention_heads: int
    hidden_dim: int
    dropout_frac: float


FeatureMapExtractorConfig = Union[DinoFeatureMapExtractorConfig, None]
SemanticTokenExtractorConfig = Union[
    SemanticNullExtractorConfig, SemanticEmbeddingMatrixConfig, SemanticSegmentExtractorConfig]
PositionEmbeddingConfig = Union[PlanarPositionEmbeddingConfig, SphericalPositionEmbeddingConfig, NullPositionEmbeddingConfig]
AggregationConfig = Union[TransformerAggregatorConfig]

ExtractorConfig = Union[
    DinoFeatureMapExtractorConfig,
    AlphaEarthExtractorConfig,
    SemanticNullExtractorConfig,
    SemanticEmbeddingMatrixConfig,
    SemanticSegmentExtractorConfig,
    SemanticLandmarkExtractorConfig,
    PanoramaSemanticLandmarkExtractorConfig,
    PanoramaProperNounExtractorConfig,
    PanoramaLocationTypeExtractorConfig,
    ComposableCLIPPanoExtractorConfig,
    ComposableCLIPOSMExtractorConfig,
    SyntheticLandmarkExtractorConfig,
    AbsolutePositionExtractorConfig,
]


class CacheableExtractorInfo(NamedTuple):
    """Information about an extractor used to find its cache.

    Passed from model to dataset to compute cache keys.
    """
    model_config: ExtractorConfig
    patch_dims: tuple[int, int]
