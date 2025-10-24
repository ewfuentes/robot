
import common.torch.load_torch_deps
import torch
import torch.nn.functional as F
import torchvision as tv
import msgspec
from pathlib import Path
from typing import Any
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, SemanticTokenExtractorOutput, FeatureMapExtractorOutput, ExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_segment_extractor import SemanticSegmentExtractor
from experimental.overhead_matching.swag.model.alphaearth_extractor import AlphaEarthExtractor
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import SemanticLandmarkExtractor
from experimental.overhead_matching.swag.model.panorama_semantic_landmark_extractor import PanoramaSemanticLandmarkExtractor
from torch.nn.init import xavier_uniform_
from experimental.overhead_matching.swag.model.synthetic_landmark_extractor import SyntheticLandmarkExtractor
from experimental.overhead_matching.swag.model.absolute_position_extractor import AbsolutePositionExtractor
from experimental.overhead_matching.swag.model.osm_semantic_class_extractor import OSMSemanticClassExtractor
from experimental.overhead_matching.swag.model.swag_config_types import (
    FeatureMapExtractorConfig,
    DinoFeatureMapExtractorConfig,
    AlphaEarthExtractorConfig,

    SemanticTokenExtractorConfig,
    SemanticNullExtractorConfig,
    SemanticEmbeddingMatrixConfig,
    SemanticSegmentExtractorConfig,
    SemanticLandmarkExtractorConfig,
    PanoramaSemanticLandmarkExtractorConfig,
    AbsolutePositionExtractorConfig,
    SyntheticLandmarkExtractorConfig,
    OSMSemanticClassExtractorConfig,

    PositionEmbeddingConfig,
    PlanarPositionEmbeddingConfig,
    SphericalPositionEmbeddingConfig,
    NullPositionEmbeddingConfig,

    AggregationConfig,
    TransformerAggregatorConfig,

    ExtractorConfig,
    ExtractorDataRequirement,
    CacheableExtractorInfo,
)


class SwagPatchEmbeddingConfig(msgspec.Struct, tag=True, tag_field="kind"):
    position_embedding_config: PositionEmbeddingConfig
    aggregation_config: AggregationConfig

    patch_dims: tuple[int, int]
    output_dim: int
    num_embeddings: int

    extractor_config_by_name: dict[str, ExtractorConfig] = {}
    use_cached_extractors: list[str] = []

    auxiliary_info: dict[str, Any] = {}

    normalize_embeddings: bool = True

    # These are here for backwards compatibility
    feature_map_extractor_config: FeatureMapExtractorConfig | None = None
    semantic_token_extractor_config: SemanticTokenExtractorConfig | None = None
    use_cached_feature_maps: bool = False
    use_cached_semantic_tokens: bool = False


def create_extractor(config: ExtractorConfig, auxiliary_info: dict[str, Any]):
    if config is None:
        return None
    match config:
        case DinoFeatureMapExtractorConfig(): return DinoFeatureExtractor(config)
        case SemanticNullExtractorConfig(): return SemanticNullExtractor(config)
        case SemanticEmbeddingMatrixConfig(): return SemanticEmbeddingMatrix(config)
        case SemanticLandmarkExtractorConfig(): return SemanticLandmarkExtractor(config, auxiliary_info[config.auxiliary_info_key])
        case PanoramaSemanticLandmarkExtractorConfig(): return PanoramaSemanticLandmarkExtractor(config, auxiliary_info[config.auxiliary_info_key])
        case SemanticSegmentExtractorConfig(): return SemanticSegmentExtractor(config)
        case SyntheticLandmarkExtractorConfig(): return SyntheticLandmarkExtractor(config)
        case AbsolutePositionExtractorConfig(): return AbsolutePositionExtractor(config)
        case AlphaEarthExtractorConfig(): return AlphaEarthExtractor(
                config, auxiliary_info[config.auxiliary_info_key])
        case OSMSemanticClassExtractorConfig(): return OSMSemanticClassExtractor(
                config, auxiliary_info[config.auxiliary_info_key])
    raise NotImplementedError(f"Unhandled Config Type: {config}")


def create_position_embedding(config: PositionEmbeddingConfig):
    match config:
        case PlanarPositionEmbeddingConfig(): return PlanarPositionEmbedding(config)
        case SphericalPositionEmbeddingConfig(): return SphericalPositionEmbedding(config)
        case NullPositionEmbeddingConfig(): return NullPositionEmbedding(config)


def create_aggregator_model(output_dim: int, config: AggregationConfig):
    match config:
        case TransformerAggregatorConfig(): return TransformerAggregator(output_dim, config)


class DinoFeatureExtractor(torch.nn.Module):
    def __init__(self, config: DinoFeatureMapExtractorConfig):
        super().__init__()
        assert config.model_str.startswith('dino')
        repo_name = f"facebookresearch/{config.model_str.split('_')[0]}"
        self.use_class_token_only = config.use_class_token_only
        self._dino = torch.hub.load(repo_name, config.model_str)
        self._dino.eval()
        for param in self._dino.parameters():
            param.requires_grad = False

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        input_image = model_input.image
        x = tv.transforms.functional.normalize(
            input_image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        # Compute the relative positions of each patch
        batch_size = len(model_input.metadata)
        original_shape = model_input.image.shape[2:]

        with torch.no_grad():
            if self.use_class_token_only:
                patch_tokens = self._dino(x).unsqueeze(1) # batch x 1 x dino_emb_dim
            else:
                patch_tokens = self._dino.forward_features(x)["x_norm_patchtokens"]

        if self.use_class_token_only:
            num_tokens = 1
            relative_positions = torch.zeros(batch_size, num_tokens, self.num_position_outputs, 2)
        else:
            num_row_tokens = original_shape[0] // self.patch_size[0]
            num_col_tokens = original_shape[1] // self.patch_size[1]
            num_tokens = num_row_tokens * num_col_tokens
            center_location = (original_shape[0] // 2, original_shape[1] // 2)

            relative_positions = torch.zeros((batch_size, num_row_tokens, num_col_tokens, 2))
            for row_idx in range(num_row_tokens):
                for col_idx in range(num_col_tokens):
                    patch_center_row_px = self.patch_size[0] // 2 + row_idx * self.patch_size[0]
                    patch_center_col_px = self.patch_size[1] // 2 + col_idx * self.patch_size[1]

                    relative_positions[:, row_idx, col_idx, 0] = (
                            patch_center_row_px - center_location[0])
                    relative_positions[:, row_idx, col_idx, 1] = (
                            patch_center_col_px - center_location[1])

            relative_positions = relative_positions.reshape(
                    batch_size, num_tokens, self.num_position_outputs, 2)
        assert num_tokens == patch_tokens.shape[1]

        mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=patch_tokens.device)

        return ExtractorOutput(
            features=patch_tokens,
            positions=relative_positions.to(device=patch_tokens.device),
            mask=mask.to(device=patch_tokens.device),
            debug={})

    @property
    def patch_size(self):
        p = self._dino.patch_size
        return (p, p)

    @property
    def output_dim(self):
        return self._dino.num_features

    @property
    def num_position_outputs(self):
        return 1

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return [ExtractorDataRequirement.IMAGES]


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
                positions=torch.zeros((batch_size, 0, self.num_position_outputs, 2), device=dev),
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
            positions=positions_in_patch.reshape(batch_size, max_num_landmarks, self.num_position_outputs, 2),
            features=self._embedding_matrix(output_idxs),
            mask=mask)

    @property
    def output_dim(self):
        return self._embedding_matrix.embedding_dim

    @property
    def num_position_outputs(self):
        return 1

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return [ExtractorDataRequirement.LANDMARKS]


class SemanticNullExtractor(torch.nn.Module):
    def __init__(self, config: SemanticNullExtractorConfig):
        super().__init__()

    def forward(self, model_input: ModelInput):
        batch_size = len(model_input.metadata)
        dev = model_input.image.device
        positions_in_patch = torch.zeros((batch_size, 0, self.num_position_outputs, 2), device=dev)
        semantic_tokens = torch.zeros((batch_size, 0, 0), device=dev)
        mask = torch.zeros(batch_size, 0, device=dev, dtype=torch.bool)
        return SemanticTokenExtractorOutput(
            positions=positions_in_patch,
            features=semantic_tokens,
            mask=mask)

    @property
    def output_dim(self):
        return 0

    @property
    def num_position_outputs(self):
        return 1

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return []


class SphericalPositionEmbedding(torch.nn.Module):
    def __init__(self, config: SphericalPositionEmbeddingConfig):
        super().__init__()
        self._embedding_dim = config.embedding_dim
        self._scale_step = config.scale_step

        assert self._embedding_dim % 4 == 0

    def forward(self, *,
                model_input: ModelInput,
                relative_positions: torch.Tensor):
        batch_size, num_tokens = relative_positions.shape[:2]
        # We need to convert the pixel positions into an elevation and azimuth angles
        # The azimuth angle increases in the clockwise direction
        # We assume that zero degrees is on the left edge of the panorama and increases
        # as we move from left to right.
        # The elevation angle is zero at the top of the image and increases as you move
        # from top to bottom. Note that these are not standard definitions of these angles
        # but it does simplify implementation, and ultimately shouldn't matter
        original_shape = model_input.image.shape[-2:]
        elevation_rad = torch.pi / 2 - relative_positions[..., 0] / original_shape[0] * torch.pi
        azimuth_rad = relative_positions[..., 1] / original_shape[1] * 2 * torch.pi
        out = torch.zeros((*relative_positions.shape[:-1], self._embedding_dim), dtype=torch.float32)

        num_scales = self._embedding_dim // 4
        for scale_idx in range(num_scales):
            embedding_idx_start = 4 * scale_idx
            scale = (self._scale_step ** scale_idx)

            out[..., embedding_idx_start + 0] = torch.sin(elevation_rad * scale)
            out[..., embedding_idx_start + 1] = torch.cos(elevation_rad * scale)
            out[..., embedding_idx_start + 2] = torch.sin(azimuth_rad * scale)
            out[..., embedding_idx_start + 3] = torch.cos(azimuth_rad * scale)
        return out.reshape(batch_size, num_tokens, -1)

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
                relative_positions: torch.Tensor):
        assert relative_positions.ndim == 4
        batch_size, num_tokens, num_position_tokens = relative_positions.shape[:-1]
        out = torch.zeros((*relative_positions.shape[:-1], self._embedding_dim), dtype=torch.float32)

        num_scales = self._embedding_dim // 4
        for scale_idx in range(num_scales):
            embedding_idx_start = 4 * scale_idx
            scale = (self._min_scale * self._scale_step ** scale_idx) / (2 * torch.pi)

            out[..., embedding_idx_start + 0] = torch.sin(relative_positions[..., 0] / scale)
            out[..., embedding_idx_start + 1] = torch.cos(relative_positions[..., 0] / scale)
            out[..., embedding_idx_start + 2] = torch.sin(relative_positions[..., 1] / scale)
            out[..., embedding_idx_start + 3] = torch.cos(relative_positions[..., 1] / scale)
        return out.reshape(batch_size, num_tokens, num_position_tokens * self._embedding_dim)

    @property
    def output_dim(self):
        return self._embedding_dim


class NullPositionEmbedding(torch.nn.Module):
    def __init__(self, config: NullPositionEmbeddingConfig):
        super().__init__()

    def forward(self, *,
                model_input: ModelInput,
                relative_positions: torch.Tensor):
        assert relative_positions.ndim == 4  # (batch, tokens, num_positions, 2)
        batch_size, num_tokens = relative_positions.shape[:2]
        out = torch.zeros((batch_size, num_tokens, 0),
                         dtype=torch.float32,
                         device=relative_positions.device)
        return out

    @property
    def output_dim(self):
        return 0


def init_xavier(model):
    for p in model.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)


def make_float_mask_from_bool_mask(bool_mask_true_means_mask: torch.Tensor) -> torch.Tensor:
    """
    Create a float mask for a transformer en/decoder from a boolean mask.
    TRUE in the input bool mask indicates we WILL mask that token.

    Args:
        bool_mask_true_means_mask: bool tensor where false allows attention and true does not 
    Returns:
        float tensor of the same shape. Zeros in all false positions, and -inf in true positions.

        These values are added to the attention before softmax, so -inf zeros out. 
    """
    assert bool_mask_true_means_mask.dtype == torch.bool, f"Expected mask to be dtype bool, got {bool_mask_true_means_mask.dtype}"
    return torch.where(bool_mask_true_means_mask, -torch.inf, 0.0)

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
        # see warning at https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        init_xavier(self._encoder)

    def forward(self, tokens, token_mask):
        float_token_mask = make_float_mask_from_bool_mask(token_mask)
        return self._encoder(tokens, src_key_padding_mask=float_token_mask, is_causal=False)


class SwagPatchEmbedding(torch.nn.Module):
    def __init__(self, config: SwagPatchEmbeddingConfig):
        super().__init__()
        self._extractor_by_name = torch.nn.ModuleDict({
            k: create_extractor(c, config.auxiliary_info)
            for k, c in config.extractor_config_by_name.items()})
        self._normalize_embeddings = config.normalize_embeddings
        self._token_marker_by_name = torch.nn.ParameterDict({
                k: torch.nn.Parameter(torch.randn(1, 1, config.output_dim))
                for k in config.extractor_config_by_name})

        self._position_embedding = create_position_embedding(
                config.position_embedding_config)

        self._projection_by_name = torch.nn.ModuleDict({
                k: torch.nn.Linear(self._extractor_by_name[k].output_dim +
                                   self._extractor_by_name[k].num_position_outputs *
                                   self._position_embedding.output_dim,
                                   config.output_dim)
                for k in config.extractor_config_by_name})

        self._cls_token = torch.nn.Parameter(torch.randn((1, config.num_embeddings, config.output_dim)))

        self._aggregator_model = create_aggregator_model(
                config.output_dim, config.aggregation_config)

        # Store cacheable extractor info
        self._cacheable_extractor_info = {}
        for k in config.use_cached_extractors:
            self._cacheable_extractor_info[k] = CacheableExtractorInfo(
                model_config=config.extractor_config_by_name[k],
                patch_dims=config.patch_dims)

        self._patch_dims = config.patch_dims
        self._output_dim = config.output_dim

        # We keep these for backwards compatibility
        self._feature_map_extractor = create_extractor(
                config.feature_map_extractor_config, config.auxiliary_info)
        self._semantic_token_extractor = create_extractor(
                config.semantic_token_extractor_config, config.auxiliary_info)
        if self._feature_map_extractor is not None:
            self._extractor_by_name["__feature_map_extractor"] = self._feature_map_extractor
            self._feature_token_marker = torch.nn.Parameter(torch.randn((1, 1, config.output_dim)))
            self._token_marker_by_name["__feature_map_extractor"] = self._feature_token_marker
            self._feature_token_projection = torch.nn.Linear(
                    self._feature_map_extractor.output_dim + self._position_embedding.output_dim,
                    config.output_dim)
            self._projection_by_name["__feature_map_extractor"] = self._feature_token_projection
            if config.use_cached_feature_maps:
                self._cacheable_extractor_info["__feature_map_extractor"] = CacheableExtractorInfo(
                    model_config=config.feature_map_extractor_config,
                    patch_dims=config.patch_dims)

        if self._semantic_token_extractor is not None:
            self._extractor_by_name["__semantic_token_extractor"] = self._semantic_token_extractor
            self._semantic_token_marker = torch.nn.Parameter(torch.randn((1, 1, config.output_dim)))
            self._token_marker_by_name["__semantic_token_extractor"] = self._semantic_token_marker
            self._semantic_token_projection = torch.nn.Linear(
                    self._semantic_token_extractor.output_dim + self._position_embedding.output_dim,
                    config.output_dim)
            self._projection_by_name["__semantic_token_extractor"] = self._semantic_token_projection

            if config.use_cached_semantic_tokens:
                self._cacheable_extractor_info["__semantic_token_extractor"] = CacheableExtractorInfo(
                    model_config=config.semantic_token_extractor_config,
                    patch_dims=config.patch_dims)

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

    def _get_input_tokens(self, model_input: ModelInput) -> tuple[torch.Tensor, torch.Tensor, dict[str, ExtractorOutput]]:
        dev = self._cls_token.device
        extractor_outputs_by_name = {}
        for k in self._extractor_by_name:
            extractor_outputs_by_name[k] = model_input.cached_tensors.get(k)
            if extractor_outputs_by_name[k] is None:
                extractor_outputs_by_name[k] = self._extractor_by_name[k](model_input)
                assert extractor_outputs_by_name[k].positions.ndim == 4, f"relative positions of {k} is not 4 dimensional"

        input_tokens_by_name = {}
        for k, v in extractor_outputs_by_name.items():
            if v.positions.shape[1] == 0:  # no features
                continue
            position_embeddings = self._position_embedding(
                model_input=model_input,
                relative_positions=v.positions).to(dev)

            feature_tokens = torch.cat([v.features, position_embeddings], dim=-1)
            feature_tokens = self._projection_by_name[k](feature_tokens)
            feature_tokens += self._token_marker_by_name[k]
            input_tokens_by_name[k] = feature_tokens

        batch_size = model_input.image.shape[0]
        cls_token = self._cls_token.expand(batch_size, -1, -1)
        input_tokens = torch.cat([cls_token] + list(input_tokens_by_name.values()), dim=1)
        if self._normalize_embeddings:
            input_tokens = F.normalize(input_tokens, dim=-1)

        cls_mask = torch.zeros(
                cls_token.shape[:2], device=dev, dtype=torch.bool)
        input_mask = torch.cat([cls_mask] +
                               [v.mask for v in extractor_outputs_by_name.values()], dim=1)

        return input_tokens, input_mask, extractor_outputs_by_name

    def forward(self, model_input: ModelInput) -> tuple[torch.Tensor, dict[str, ExtractorOutput]]:
        """Forward pass through the model.

        Args:
            model_input: Input containing image and metadata

        Returns:
            Tuple of (embeddings, extractor_outputs_by_name) where:
                - embeddings: Tensor of shape (batch, num_embeddings, output_dim)
                - debug dict: currently extractor_outputs_by_name: Dict mapping extractor names to their ExtractorOutput objects
        """
        input_tokens, input_mask, extractor_outputs_by_name = self._get_input_tokens(model_input)
        output_tokens = self._aggregator_model(input_tokens, input_mask)
        model_output = output_tokens[:, :self._cls_token.shape[1], :]  # B, num_class_tokens, D_emb

        # output is batch x num_class_tokens x feature_dim
        if self._normalize_embeddings:
            model_output = F.normalize(model_output, dim=2)

        return model_output, extractor_outputs_by_name

    def cache_info(self) -> dict[str, CacheableExtractorInfo]:
        """Returns information about cacheable extractors
        """
        return self._cacheable_extractor_info

    @property
    def patch_dims(self):
        return self._patch_dims

    @property
    def output_dim(self):
        return self._output_dim
    
    @property
    def num_embeddings(self):
        return self._cls_token.shape[1]

