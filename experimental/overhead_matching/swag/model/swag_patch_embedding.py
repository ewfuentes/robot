
import common.torch.load_torch_deps
import torch
import torchvision as tv
import msgspec
from typing import Union
from enum import StrEnum, auto
from dataclasses import dataclass


class FeatureMapExtractorType(StrEnum):
    DINOV2 = auto()


class DinoFeatureMapExtractorConfig(msgspec.Struct, tag=True, tag_field="kind"):
    model_str: str = "dinov2_vitb14"
    type: FeatureMapExtractorType = FeatureMapExtractorType.DINOV2


class SemanticTokenExtractorType(StrEnum):
    EMBEDDING_MAT = auto()


class SemanticEmbeddingMatrixConfig(msgspec.Struct, tag=True, tag_field="kind"):
    vocabulary: list[str]
    embedding_dim: int
    type: SemanticTokenExtractorType = SemanticTokenExtractorType.EMBEDDING_MAT


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
SemanticTokenExtractorConfig = Union[SemanticEmbeddingMatrixConfig]
PositionEmbeddingConfig = Union[PlanarPositionEmbeddingConfig, SphericalEmbeddingConfig]
AggregationConfig = Union[TransformerAggregatorConfig]


@dataclass
class SwagPatchEmbeddingConfig:
    feature_map_extractor_config: FeatureMapExtractorConfig
    semantic_token_extractor_config: SemanticTokenExtractorConfig
    position_embedding_config: PositionEmbeddingConfig
    aggregation_config: AggregationConfig

    image_input_dim: tuple[int, int]
    output_dim: int


@dataclass
class ModelInput:
    image: torch.Tensor
    metadata: list[dict]


def create_feature_map_extractor(config: FeatureMapExtractorConfig):
    assert config.type == FeatureMapExtractorType.DINOV2
    return DinoFeatureExtractor(config)


def create_semantic_token_extractor(config: SemanticTokenExtractorConfig):
    assert config.type == SemanticTokenExtractorType.EMBEDDING_MAT
    return SemanticEmbeddingMatrix(config)


def create_position_embedding(config: PositionEmbeddingConfig):
    assert config.type == PositionEmbeddingType.PLANAR
    return PlanarPositionEmbedding(config)


def create_aggregator_model(output_dim: int, config: AggregationConfig):
    assert config.type == AggregationType.TRANSFORMER
    return TransformerAggregator(output_dim, config)
    ...


class DinoFeatureExtractor(torch.nn.Module):
    def __init__(self, config: DinoFeatureMapExtractorConfig):
        super().__init__()
        self._dino = torch.hub.load("facebookresearch/dinov2", config.model_str)
        self._dino.eval()

    def forward(self, model_input: ModelInput):
        input_image = model_input.image
        x = tv.transforms.functional.normalize(
            input_image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        with torch.no_grad():
            patch_tokens = self._dino.forward_features(x)["x_norm_patchtokens"]
        return patch_tokens

    @property
    def patch_size(self):
        return (14, 14)

    @property
    def output_dim(self):
        return 768


class SemanticEmbeddingMatrix(torch.nn.Module):
    def __init__(self, config: SemanticEmbeddingMatrixConfig):
        super().__init__()
        self._vocabulary = config.vocabulary
        self._idx_from_vocab = {v: i+1 for i, v in enumerate(self._vocabulary)}
        self._embedding_matrix = torch.nn.Embedding(len(self._vocabulary) + 1,
                                                    config.embedding_dim,
                                                    padding_idx=0,
                                                    max_norm=1.0)

    def forward(self, model_input: ModelInput):
        batch_size = len(model_input.metadata)
        max_num_landmarks = max([len(x["landmarks"]) for x in model_input.metadata])
        output_idxs = torch.full((batch_size, max_num_landmarks), 0, dtype=torch.int)
        positions_in_patch = torch.zeros((batch_size, max_num_landmarks, 2), dtype=torch.float32)

        for batch_idx in range(len(model_input.metadata)):
            sat_metadata = model_input.metadata[batch_idx]

            for landmark_idx, landmark in enumerate(sat_metadata["landmarks"]):
                output_idxs[batch_idx, landmark_idx] = (
                        self._idx_from_vocab[landmark["landmark_type"]])
                positions_in_patch[batch_idx, landmark_idx, 0] = (
                        landmark["web_mercator_y"] - sat_metadata["web_mercator_y"])
                positions_in_patch[batch_idx, landmark_idx, 1] = (
                        landmark["web_mercator_x"] - sat_metadata["web_mercator_x"])
        mask = output_idxs == 0

        return positions_in_patch, self._embedding_matrix(output_idxs), mask

    @property
    def output_dim(self):
        return self._embedding_matrix.embedding_dim


class PlanarPositionEmbedding(torch.nn.Module):
    def __init__(self, config: PlanarPositionEmbeddingConfig):
        super().__init__()
        self._embedding_dim = config.embedding_dim
        self._min_scale = config.min_scale
        self._scale_step = config.scale_step

        assert self._embedding_dim % 4 == 0

    def forward(self, *,
                relative_positions: None | torch.Tensor = None,
                model_input: None | ModelInput = None,
                patch_size: None | tuple[int, int] = None):

        # If we're being given the model input and the patch size, we have to compute the
        # relative patch locations
        if model_input is not None and patch_size is not None:
            batch_size = len(model_input.metadata)
            original_size = model_input.metadata[0]["original_size"]

            num_row_tokens = original_size[0] // patch_size[0]
            num_col_tokens = original_size[1] // patch_size[1]
            num_tokens = num_row_tokens * num_col_tokens

            center_location = (original_size[0] // 2, original_size[1] // 2)

            relative_positions = torch.zeros((batch_size, num_row_tokens, num_col_tokens, 2))
            for row_idx in range(num_row_tokens):
                for col_idx in range(num_col_tokens):
                    patch_center_row_px = patch_size[0] // 2 + row_idx * patch_size[0]
                    patch_center_col_px = patch_size[1] // 2 + col_idx * patch_size[1]

                    relative_positions[:, row_idx, col_idx, 0] = (
                            patch_center_row_px - center_location[0])
                    relative_positions[:, row_idx, col_idx, 1] = (
                            patch_center_col_px - center_location[1])

            relative_positions = relative_positions.reshape(batch_size, num_tokens, 2)

        batch_size, num_tokens = relative_positions.shape[:2]

        out = torch.zeros((batch_size, num_tokens, self._embedding_dim), dtype=torch.float32)

        num_scales = self._embedding_dim // 4
        for scale_idx in range(num_scales):
            embedding_idx_start = 4 * scale_idx
            scale = self._min_scale * self._scale_step ** scale_idx / (2 * torch.pi)

            out[..., embedding_idx_start + 0] = torch.sin(relative_positions[..., 0] / scale)
            out[..., embedding_idx_start + 1] = torch.cos(relative_positions[..., 0] / scale)
            out[..., embedding_idx_start + 2] = torch.sin(relative_positions[..., 1] / scale)
            out[..., embedding_idx_start + 3] = torch.cos(relative_positions[..., 1] / scale)
        return out

    @property
    def output_dim(self):
        return self._embedding_dim


class TransformerAggregator(torch.nn.Module):
    def __init__(self, output_dim: int, config: TransformerAggregatorConfig):
        super().__init__()
        transformer_layer = torch.nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout_frac,
                batch_first=True)

        self._encoder = torch.nn.TransformerEncoder(
                transformer_layer,
                num_layers=config.num_transformer_layers)

    def forward(self, tokens, token_mask):
        return self._encoder(tokens, src_key_padding_mask=token_mask, is_causal=False)


class SwagPatchEmbedding(torch.nn.Module):
    def __init__(self, config: SwagPatchEmbeddingConfig):
        super().__init__()
        self._feature_map_extractor = create_feature_map_extractor(
                config.feature_map_extractor_config)
        self._semantic_token_extractor = create_semantic_token_extractor(
                config.semantic_token_extractor_config)
        self._position_embedding = create_position_embedding(
                config.position_embedding_config)
        self._aggregator_model = create_aggregator_model(
                config.output_dim, config.aggregation_config)

        self._feature_token_marker = torch.nn.Parameter(torch.randn((1, 1, config.output_dim)))
        self._semantic_token_marker = torch.nn.Parameter(torch.randn((1, 1, config.output_dim)))
        self._cls_token = torch.nn.Parameter(torch.randn((1, 1, config.output_dim)))

        self._feature_token_projection = torch.nn.Linear(
                self._feature_map_extractor.output_dim + self._position_embedding.output_dim,
                config.output_dim)
        self._semantic_token_projection = torch.nn.Linear(
                self._semantic_token_extractor.output_dim + self._position_embedding.output_dim,
                config.output_dim)

    def forward(self, model_input: ModelInput):
        # feature positions is batch x n_tokens x 2
        # feature amp is batch x n_tokens x feature_dim
        # The positions are where each token comes from in pixel space in the original image
        feature_map = self._feature_map_extractor(model_input)
        # semantic positions is batch x n_semantic_tokens x 2
        # semantic tokens is batch x n_semantic_tokens x semantic_feature_dim
        # The positions are where each token comes from in pixel space in the original image
        # The semantic_mask is true if the corresponding token is a padding token
        semantic_positions, semantic_tokens, semantic_mask = (
                self._semantic_token_extractor(model_input))

        # feature_position embeddings is batch x n_tokens x pos_embedding_dim
        # semantic_position_embeddings is batch x n_sematic_tokens x pos_embedding_dim
        feature_position_embeddings = self._position_embedding(
                model_input=model_input,
                patch_size=self._feature_map_extractor.patch_size)
        semantic_position_embeddings = self._position_embedding(
                relative_positions=semantic_positions)

        pos_feature_tokens = torch.cat([feature_map, feature_position_embeddings], dim=-1)
        pos_semantic_tokens = torch.cat([semantic_tokens, semantic_position_embeddings], dim=-1)

        input_feature_tokens = self._feature_token_projection(pos_feature_tokens)
        input_semantic_tokens = self._semantic_token_projection(pos_semantic_tokens)

        input_feature_tokens += self._feature_token_marker
        input_semantic_tokens += self._semantic_token_marker

        # input tokens is batch x (n_semantic_tokens + n_feature_tokens + 1) x aggregation_dim
        batch_size = input_feature_tokens.shape[0]
        cls_token = self._cls_token.expand(batch_size, -1, -1)
        input_tokens = torch.cat([cls_token, input_feature_tokens, input_semantic_tokens], dim=1)

        feature_and_cls_mask = torch.zeros((batch_size, input_feature_tokens.shape[1] + 1))
        input_mask = torch.cat([feature_and_cls_mask, semantic_mask], dim=1)

        output_tokens = self._aggregator_model(input_tokens, input_mask)

        # output is batch x feature_dim
        return output_tokens[:, 0, :]
