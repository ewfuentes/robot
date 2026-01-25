
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


class OSMFieldExtractorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for extracting embeddings for a specific OSM field from v2.0 pickle.

    This extractor works with the v2.0 pickle format that stores separate embeddings
    for different fields (name, brand, amenity, street_address, osm_sentences).

    Attributes:
        tag: Which field to extract embeddings for. Options:
            - "name": Business/place names
            - "brand": Brand names
            - "amenity": Amenity types
            - "street_address": Street addresses
            - "osm_sentences": Full OSM sentence descriptions
        openai_embedding_size: Output embedding dimension (crops if smaller than source)
        embedding_version: Version directory containing embeddings.pkl
        auxiliary_info_key: Key in auxiliary_info dict for base path
        landmark_type: Optional geometry filter (point, linestring, polygon, multipolygon)
    """
    tag: str
    openai_embedding_size: int
    embedding_version: str
    auxiliary_info_key: str
    landmark_type: LandmarkType | None = None


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
    OSMFieldExtractorConfig,
    SyntheticLandmarkExtractorConfig,
    AbsolutePositionExtractorConfig,
]


class CacheableExtractorInfo(NamedTuple):
    """Information about an extractor used to find its cache.

    Passed from model to dataset to compute cache keys.
    """
    model_config: ExtractorConfig
    patch_dims: tuple[int, int]
