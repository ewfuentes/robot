
import common.torch.load_torch_deps
import torch
import msgspec
from typing import Union
from enum import StrEnum, auto
from dataclasses import dataclass


class FeatureMapExtractorType(StrEnum):
    DINOV2 = auto()


class DinoFeatureMapExtractorConfig(msgspec.Struct, tag=True, tag_field="kind"):
    model: str = "dinov2_vitb14"
    type: FeatureMapExtractorType = FeatureMapExtractorType.DINOV2


class SemanticTokenExtractorType(StrEnum):
    EMBEDDING_MAT = auto()


class EmbeddingMatrixSemanticTokenExtractor(msgspec.Struct, tag=True, tag_field="kind"):
    vocabulary: list[str]
    type: SemanticTokenExtractorType = SemanticTokenExtractorType.EMBEDDING_MAT


class PositionEmbeddingType(StrEnum):
    GPS = auto()
    SPHERICAL = auto()


class GPSEmbeddingConfig(msgspec.Struct, tag=True, tag_field="kind"):
    type: PositionEmbeddingType = PositionEmbeddingType.GPS


class SphericalEmbeddingConfig(msgspec.Struct, tag=True, tag_field="kind"):
    type: PositionEmbeddingType = PositionEmbeddingType.SPHERICAL


class AggregationType(StrEnum):
    TRANSFORMER = auto()


class TransformerAggregatorConfig(msgspec.Struct, tag=True, tag_field="kind"):
    type: AggregationType = AggregationType.TRANSFORMER


FeatureMapExtractorConfig = Union[DinoFeatureMapExtractorConfig]
SemanticTokenExtractorConfig = Union[EmbeddingMatrixSemanticTokenExtractor]
PositionEmbeddingConfig = Union[GPSEmbeddingConfig, SphericalEmbeddingConfig]
AggregationConfig = Union[TransformerAggregatorConfig]


@dataclass
class SwagPatchEmbeddingConfig:
    feature_map_extractor_config: FeatureMapExtractorConfig
    semantic_token_extractor_config: SemanticTokenExtractorConfig
    position_embedding_config: PositionEmbeddingConfig
    aggregation_config: AggregationConfig

    image_input_dim: tuple[int, int]
    position_embedding_dim: int
    output_dim: int


class SwagPatchEmbedding(torch.nn.Module):
    def __init__(self, config: SwagPatchEmbeddingConfig):
        self._feature_map_extractor = create_feature_map_extractor(
                config.feature_map_extractor_config)
        self._semantic_token_extractor = create_semantic_token_extractor(
                config.semantic_token_extractor_config)
        self._position_embedding = create_position_embedding(
                config.position_embedding_config)
        self._aggregator_model = create_aggregator_model(
                config.aggregation_config)

        self._feature_token_marker = torch.nn.Parameter(torch.randn((1, 1, config.output_dim)))
        self._semantic_token_marker = torch.nn.Parameter(torch.randn((1, 1, config.output_dim)))
        self._cls_token = torch.nn.Parameter(torch.randn((1, 1, config.output_dim)))

    def forward(self, model_input):
        # feature positions is batch x n_tokens x 2
        # feature amp is batch x n_tokens x feature_dim
        feature_positions, feature_map = self._feature_map_extractor(model_input)
        # semantic positions is batch x n_semantic_tokens x 2
        # semantic tokens is batch x n_semantic_tokens x semantic_feature_dim
        semantic_positions, semantic_tokens = self._semantic_token_extractor(model_input)

        # feature_position embeddings is batch x n_tokens x pos_embedding_dim
        # semantic_position_embeddings is batch x n_sematic_tokens x pos_embedding_dim
        feature_position_embeddings, semantic_position_embeddings = self._position_embedding(
                feature_positions, semantic_positions)

        pos_feature_tokens = torch.cat([feature_map, feature_position_embeddings], dim=-1)
        pos_semantic_tokens = torch.cat([semantic_tokens, semantic_position_embeddings], dim=-1)

        input_feature_tokens = self._feature_token_projection(pos_feature_tokens)
        input_semantic_tokens = self._semantic_token_projection(pos_semantic_tokens)

        input_feature_tokens += self._feature_token_marker
        input_semantic_tokens += self._semantic_token_marker

        # input tokens is batch x (n_semantic_tokens + n_feature_tokens + 1) x aggregation_dim
        input_tokens = torch.cat([self._cls_token, input_feature_tokens, input_semantic_tokens])

        output_tokens = self._aggregator_model(input_tokens)

        # output is batch x feature_dim
        return output_tokens[:, 0, :]
