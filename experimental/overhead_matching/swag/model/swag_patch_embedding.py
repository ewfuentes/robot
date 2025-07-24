
import common.torch.load_torch_deps
import torch
import torch.nn.functional as F
import torchvision as tv
import msgspec
import hashlib
from typing import Any
from dataclasses import dataclass
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, SemanticTokenExtractorOutput, FeatureMapExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_segment_extractor import SemanticSegmentExtractor
from experimental.overhead_matching.swag.model.swag_config_types import (
    FeatureMapExtractorConfig,
    FeatureMapExtractorType,
    DinoFeatureMapExtractorConfig,

    SemanticTokenExtractorConfig,
    SemanticTokenExtractorType,
    SemanticNullExtractorConfig,
    SemanticEmbeddingMatrixConfig,
    SemanticSegmentExtractorConfig,

    PositionEmbeddingConfig,
    PositionEmbeddingType,
    PlanarPositionEmbeddingConfig,
    SphericalPositionEmbeddingConfig,

    AggregationConfig,
    AggregationType,
    TransformerAggregatorConfig
)


class HashStruct(msgspec.Struct, frozen=True):
    model_config: Any
    patch_dims: tuple[int, int]


def compute_config_hash(obj):
    yaml_str = msgspec.yaml.encode(obj, order='deterministic')
    return hashlib.sha256(yaml_str).hexdigest()


class SwagPatchEmbeddingConfig(msgspec.Struct, tag=True, tag_field="kind"):
    feature_map_extractor_config: FeatureMapExtractorConfig
    semantic_token_extractor_config: SemanticTokenExtractorConfig
    position_embedding_config: PositionEmbeddingConfig
    aggregation_config: AggregationConfig

    patch_dims: tuple[int, int]
    output_dim: int
    use_cached_feature_maps: bool = False
    use_cached_semantic_tokens: bool = False


def create_feature_map_extractor(config: FeatureMapExtractorConfig):
    types = {
        FeatureMapExtractorType.DINOV2: DinoFeatureExtractor
    }
    assert config.type == FeatureMapExtractorType.DINOV2
    return types[config.type](config)


def create_semantic_token_extractor(config: SemanticTokenExtractorConfig):
    types = {
        SemanticTokenExtractorType.NULL_EXTRACTOR: SemanticNullExtractor,
        SemanticTokenExtractorType.EMBEDDING_MAT: SemanticEmbeddingMatrix,
        SemanticTokenExtractorType.SEGMENT_EXTRACTOR: SemanticSegmentExtractor,
    }
    return types[config.type](config)


def create_position_embedding(config: PositionEmbeddingConfig):
    types = {
        PositionEmbeddingType.PLANAR: PlanarPositionEmbedding,
        PositionEmbeddingType.SPHERICAL: SphericalPositionEmbedding,
    }
    return types[config.type](config)


def create_aggregator_model(output_dim: int, config: AggregationConfig):
    types = {
        AggregationType.TRANSFORMER: TransformerAggregator,
    }
    return types[config.type](output_dim, config)


class DinoFeatureExtractor(torch.nn.Module):
    def __init__(self, config: DinoFeatureMapExtractorConfig):
        super().__init__()
        self._dino = torch.hub.load("facebookresearch/dinov2", config.model_str)
        self._dino.eval()

    def forward(self, model_input: ModelInput) -> FeatureMapExtractorOutput:
        input_image = model_input.image
        x = tv.transforms.functional.normalize(
            input_image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        with torch.no_grad():
            patch_tokens = self._dino.forward_features(x)["x_norm_patchtokens"]
        return FeatureMapExtractorOutput(features=patch_tokens)

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

    def forward(self, model_input: ModelInput) -> SemanticTokenExtractorOutput:
        batch_size = len(model_input.metadata)
        max_num_landmarks = max([len(x["landmarks"]) for x in model_input.metadata])
        output_idxs = torch.full((batch_size, max_num_landmarks), 0, dtype=torch.int)
        positions_in_patch = torch.zeros((batch_size, max_num_landmarks, 2), dtype=torch.float32)

        dev = self._embedding_matrix.weight.device

        if max_num_landmarks == 0:
            return SemanticTokenExtractorOutput(
                positions=torch.zeros((batch_size, 0, 2), device=dev),
                features=torch.zeros((batch_size, 0, self._embedding_matrix.embedding_dim), device=dev),
                mask=torch.zeros((batch_size, 0), device=dev))

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

        mask = mask.to(dev)
        output_idxs = output_idxs.to(dev)

        return SemanticTokenExtractorOutput(
            positions=positions_in_patch,
            features=self._embedding_matrix(output_idxs),
            mask=mask)

    @property
    def output_dim(self):
        return self._embedding_matrix.embedding_dim


class SemanticNullExtractor(torch.nn.Module):
    def __init__(self, config: SemanticNullExtractorConfig):
        super().__init__()

    def forward(self, model_input: ModelInput):
        batch_size = len(model_input.metadata)
        dev = model_input.image.device
        positions_in_patch = torch.zeros((batch_size, 0, 2), device=dev)
        semantic_tokens = torch.zeros((batch_size, 0, 0), device=dev)
        mask = torch.zeros(batch_size, 0, device=dev)
        return SemanticTokenExtractorOutput(
            positions=positions_in_patch,
            features=semantic_tokens,
            mask=mask)

    @property
    def output_dim(self):
        return 0


class SphericalPositionEmbedding(torch.nn.Module):
    def __init__(self, config: SphericalPositionEmbeddingConfig):
        super().__init__()
        self._embedding_dim = config.embedding_dim
        self._scale_step = config.scale_step

        assert self._embedding_dim % 4 == 0

    def forward(self, *,
                model_input: ModelInput,
                relative_positions: None | torch.Tensor = None,
                patch_size: None | tuple[int, int] = None):
        # If we're being given the patch size, we're being asked to compute the
        # pixel coordinates of a feature map that has the given patch size, assuming
        # no overlap
        original_shape = model_input.image.shape[-2:]
        if patch_size is not None:
            batch_size = len(model_input.metadata)

            num_row_tokens = original_shape[0] // patch_size[0]
            num_col_tokens = original_shape[1] // patch_size[1]
            num_tokens = num_row_tokens * num_col_tokens

            center_location = (original_shape[0] // 2, original_shape[1] // 2)

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

        # We need to convert the pixel positions into an elevation and azimuth angles
        # The azimuth angle increases in the clockwise direction
        # We assume that zero degrees is on the left edge of the panorama and increases
        # as we move from left to right.
        # The elevation angle is zero at the top of the image and increases as you move
        # from top to bottom. Note that these are not standard definitions of these angles
        # but it does simplify implementation, and ultimately shouldn't matter
        elevation_rad = relative_positions[..., 0] / original_shape[0] * torch.pi
        azimuth_rad = relative_positions[..., 1] / original_shape[1] * 2 * torch.pi
        out = torch.zeros((*relative_positions.shape[:-1], self._embedding_dim), dtype=torch.float32)

        num_scales = self._embedding_dim // 4
        for scale_idx in range(num_scales):
            embedding_idx_start = 4 * scale_idx
            scale = (self._scale_step ** -scale_idx) / (2 * torch.pi)

            out[..., embedding_idx_start + 0] = torch.sin(elevation_rad / scale)
            out[..., embedding_idx_start + 1] = torch.cos(elevation_rad / scale)
            out[..., embedding_idx_start + 2] = torch.sin(azimuth_rad / scale)
            out[..., embedding_idx_start + 3] = torch.cos(azimuth_rad / scale)
        return out

    @property
    def output_dim(self):
        return self._embedding_dim


class PlanarPositionEmbedding(torch.nn.Module):
    def __init__(self, config: PlanarPositionEmbeddingConfig):
        super().__init__()
        self._embedding_dim = config.embedding_dim
        self._min_scale = config.min_scale
        self._scale_step = config.scale_step

        assert self._embedding_dim % 4 == 0

    def forward(self, *,
                model_input: ModelInput,
                relative_positions: None | torch.Tensor = None,
                patch_size: None | tuple[int, int] = None):

        # If we're being given the model input and the patch size, we have to compute the
        # relative patch locations
        if patch_size is not None:
            batch_size = len(model_input.metadata)
            original_shape = model_input.image.shape[2:]

            num_row_tokens = original_shape[0] // patch_size[0]
            num_col_tokens = original_shape[1] // patch_size[1]
            num_tokens = num_row_tokens * num_col_tokens

            center_location = (original_shape[0] // 2, original_shape[1] // 2)

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
            scale = (self._min_scale * self._scale_step ** scale_idx) / (2 * torch.pi)

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
                enable_nested_tensor=False,
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

        self._patch_dims = config.patch_dims
        self._output_dim = config.output_dim

        self._cache_info = {}
        if config.use_cached_feature_maps:
            config_hash = compute_config_hash(HashStruct(
                model_config=config.feature_map_extractor_config, patch_dims=config.patch_dims))
            self._cache_info[config_hash] = ("feature_maps", FeatureMapExtractorOutput)
        if config.use_cached_semantic_tokens:
            config_hash = compute_config_hash(HashStruct(
                model_config=config.semantic_token_extractor_config, patch_dims=config.patch_dims))
            self._cache_info[config_hash] = ("semantic_tokens", SemanticTokenExtractorOutput)

    def model_input_from_batch(self, batch_item):
        if self._patch_dims[0] != self._patch_dims[1]:
            return ModelInput(
                image=batch_item.panorama,
                metadata=batch_item.panorama_metadata,
                cached_tensors=batch_item.cached_panorama_tensors)
        else:
            return ModelInput(
                image=batch_item.satellite,
                metadata=batch_item.satellite_metadata,
                cached_tensors=batch_item.cached_satellite_tensors)

    def forward(self, model_input: ModelInput):
        dev = model_input.image.device
        # feature positions is batch x n_tokens x 2
        # feature amp is batch x n_tokens x feature_dim
        # The positions are where each token comes from in pixel space in the original image
        feature_map_output = model_input.cached_tensors.get('feature_maps')
        if feature_map_output is None:
            feature_map_output = self._feature_map_extractor(model_input)
        # semantic positions is batch x n_semantic_tokens x 2
        # semantic tokens is batch x n_semantic_tokens x semantic_feature_dim
        # The positions are where each token comes from in pixel space in the original image
        # The semantic_mask is true if the corresponding token is a padding token
        semantic_tokens_output = model_input.cached_tensors.get("semantic_tokens")
        if semantic_tokens_output is None:
            semantic_tokens_output = self._semantic_token_extractor(model_input)

        # feature_position embeddings is batch x n_tokens x pos_embedding_dim
        # semantic_position_embeddings is batch x n_sematic_tokens x pos_embedding_dim
        feature_position_embeddings = self._position_embedding(
                model_input=model_input,
                patch_size=self._feature_map_extractor.patch_size).to(dev)
        semantic_position_embeddings = self._position_embedding(
                model_input=model_input,
                relative_positions=semantic_tokens_output.positions).to(dev)

        pos_feature_tokens = torch.cat([feature_map_output.features, feature_position_embeddings], dim=-1)
        pos_semantic_tokens = torch.cat([semantic_tokens_output.features, semantic_position_embeddings], dim=-1)

        input_feature_tokens = self._feature_token_projection(pos_feature_tokens)
        input_semantic_tokens = self._semantic_token_projection(pos_semantic_tokens)

        input_feature_tokens += self._feature_token_marker
        input_semantic_tokens += self._semantic_token_marker

        # input tokens is batch x (n_semantic_tokens + n_feature_tokens + 1) x aggregation_dim
        batch_size = input_feature_tokens.shape[0]
        cls_token = self._cls_token.expand(batch_size, -1, -1)
        input_tokens = torch.cat([cls_token, input_feature_tokens, input_semantic_tokens], dim=1)
        input_tokens = F.normalize(input_tokens)

        feature_and_cls_mask = torch.zeros(
                (batch_size, input_feature_tokens.shape[1] + 1), device=dev)
        input_mask = torch.cat([feature_and_cls_mask, semantic_tokens_output.mask], dim=1)

        output_tokens = self._aggregator_model(input_tokens, input_mask)

        # output is batch x feature_dim
        return F.normalize(output_tokens[:, 0, :])

    def cache_info(self):
        return self._cache_info

    @property
    def patch_dims(self):
        return self._patch_dims

    @property
    def output_dim(self):
        return self._output_dim
