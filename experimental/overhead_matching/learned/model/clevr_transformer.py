import common.torch.load_torch_deps
import torch
import enum
import numpy as np

from dataclasses import dataclass

from experimental.overhead_matching.learned.model.pose_optimizer import PoseOptimizerLayer
from experimental.overhead_matching.learned.model import clevr_tokenizer
import torchvision.transforms.v2 as tf



class InferenceMethod(enum.Enum):
    MEAN = enum.auto()
    HISTOGRAM = enum.auto()
    LEARNED_CORRESPONDENCE = enum.auto()
    OPTIMIZED_POSE = enum.auto()


@dataclass
class ClevrTransformerConfig:
    token_dim: int
    num_encoder_heads: int
    num_encoder_layers: int
    num_decoder_heads: int
    num_decoder_layers: int
    output_dim: int
    inference_method: InferenceMethod
    ego_image_tokenizer_config: clevr_tokenizer.ImageToTokensConfig | None
    overhead_image_tokenizer_config: clevr_tokenizer.ImageToTokensConfig | None


@dataclass
class SceneDescription:
    overhead_image: torch.Tensor | None
    ego_image: torch.Tensor | None
    ego_scene_description: list | None
    overhead_scene_description: list | None


def positions_from_scene_descriptions(scene_descriptions):
    batch_size = len(scene_descriptions)
    max_num_objects = max([len(x) for x in scene_descriptions])
    out = torch.zeros((batch_size, max_num_objects, 2))
    for scene_idx, scene in enumerate(scene_descriptions):
        for obj_idx, obj in enumerate(scene):
            out[scene_idx, obj_idx, :] = torch.tensor(obj["3d_coords"][:2])
    return out


class ClevrTransformer(torch.nn.Module):
    def __init__(self, config: ClevrTransformerConfig, vocabulary: dict):
        super().__init__()

        self._inference_method = config.inference_method
        self.vocabulary = vocabulary

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.num_encoder_heads,
            batch_first=True,
            dropout=0.1,
        )
        self._encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=config.token_dim,
            nhead=config.num_decoder_heads,
            batch_first=True,
            dropout=0.1,
        )

        self._decoder = torch.nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers
        )
        self._overhead_marker = torch.nn.Parameter(torch.randn(config.token_dim))
        self._ego_marker = torch.nn.Parameter(torch.randn(config.token_dim))

        self.image_preprocessing_transform = torch.nn.Sequential(
            tf.ToImage(),
            tf.ToDtype(torch.float32, scale=True),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

        if config.ego_image_tokenizer_config is not None:
            self._ego_image_tokenizer = clevr_tokenizer.ImageToTokens(
                config.ego_image_tokenizer_config)

        if config.overhead_image_tokenizer_config is not None:
            self._overhead_image_tokenizer = clevr_tokenizer.ImageToTokens(
                config.overhead_image_tokenizer_config)

        if self._inference_method == InferenceMethod.MEAN:
            self._mean_token = torch.nn.Parameter(torch.randn(config.token_dim))

        self._ego_vector_embedding = torch.nn.Embedding(
            num_embeddings=self._vocabulary_size, embedding_dim=config.token_dim
        )

        self._overhead_vector_embedding = torch.nn.Embedding(
            num_embeddings=self._vocabulary_size, embedding_dim=config.token_dim
        )

        self._output_layer = torch.nn.Linear(config.token_dim, config.output_dim)

        self._correspondence_no_match_token = torch.nn.Parameter(torch.randn(config.token_dim))

        self._pose_optimizer = PoseOptimizerLayer()

    @property
    def _vocabulary_size(self):
        return int(np.prod([len(x) for x in self.vocabulary.values()]))
    @property
    def token_dim(self):
        return self._overhead_marker.shape[0]

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_learned_correspondence(self, embedded_tokens, overhead_mask, ego_mask):
        # the embedded tokens are a batch x (n_oh + n_ego) x feature dim
        # Split the token dimension so we end up with the overhead and ego tokens
        # so we end up with a batch x  n_oh x feature dim and batch x n_ego x feature_dim
        # matrices. The masks are batch x n_oh

        batch_size = overhead_mask.shape[0]
        num_overhead_tokens = overhead_mask.shape[1]
        num_ego_tokens = ego_mask.shape[1]
        overhead_tokens = embedded_tokens[:, :num_overhead_tokens, :]
        ego_tokens = embedded_tokens[:, num_overhead_tokens:, :]

        # Add the no match token to the ego tensor so we end up with a
        # batch x (n_ego + 1) x feature_dim tensor.
        no_match_token = self._correspondence_no_match_token.expand(batch_size, 1, -1)
        ego_w_no_match_tokens = torch.cat([ego_tokens, no_match_token], dim=1)

        # Perform a batch matrix multiply so we end up with a batch x n_oh x (n_ego + 1)
        # tensor.
        ego_w_no_match_tokens = ego_w_no_match_tokens.transpose(1, 2)
        attention_logits = torch.bmm(overhead_tokens, ego_w_no_match_tokens)

        # Perform a soft max over the last dimension. This is forms our attention mask
        valid_overhead_mask = overhead_mask.to(torch.float32).unsqueeze(-1)
        valid_ego_mask = ego_mask.to(torch.float32).unsqueeze(-2)

        softmax_mask = torch.zeros((batch_size, num_overhead_tokens,
                                   num_ego_tokens+1), device=embedded_tokens.device)
        softmax_mask[..., :-1] = torch.logical_or(softmax_mask[..., :-1], valid_overhead_mask)
        softmax_mask[..., :-1] = torch.logical_or(softmax_mask[..., :-1], valid_ego_mask)
        softmax_mask[softmax_mask == True] = -torch.inf

        return torch.softmax(attention_logits + softmax_mask, dim=-1)

    def inference_mean(self, embedded_tokens, mask):
        batch_size = embedded_tokens.shape[0]
        query_tokens = self._mean_token.expand(batch_size, 1, -1)
        query_mask = None

        output_tokens = self._decoder(
            tgt=query_tokens,
            tgt_key_padding_mask=query_mask,
            memory=embedded_tokens,
            memory_key_padding_mask=mask,
            tgt_is_causal=False,
            memory_is_causal=False,
        )
        predicted_pose = self._output_layer(output_tokens)
        return predicted_pose.squeeze(1)

    def inference_histogram(self, embedded_tokens, mask, query_tokens, query_mask):
        output_tokens = self._decoder(
            tgt=query_tokens,
            tgt_key_padding_mask=query_mask,
            memory=embedded_tokens,
            memory_key_padding_mask=mask,
            tgt_is_causal=False,
            memory_is_causal=False,
        )
        return self._output_layer(output_tokens)

    def inference_learned_correspondence(self, embedded_tokens, overhead_mask, ego_mask, obj_in_overhead, obj_in_ego):
        learned_correspondence = self.compute_learned_correspondence(
            embedded_tokens, overhead_mask, ego_mask)

        optimal_pose = (None if self.training else
                        self._pose_optimizer(learned_correspondence, obj_in_overhead, obj_in_ego))
        return learned_correspondence, optimal_pose

    def inference_optimized_pose(self, embedded_tokens, overhead_mask, ego_mask, obj_in_overhead, obj_in_ego):
        learned_correspondence = self.compute_learned_correspondence(
            embedded_tokens, overhead_mask, ego_mask)

        optimal_pose = self._pose_optimizer(learned_correspondence, obj_in_overhead, obj_in_ego)
        return learned_correspondence, optimal_pose

    def forward(
        self,
        input: SceneDescription,
        query_tokens: None | torch.Tensor,
        query_mask: None | torch.Tensor,
    ):

        # tokenize overhead information
        overhead_tokens = []
        overhead_masks = []
        if input.overhead_scene_description is not None:
            overhead_result = clevr_tokenizer.create_scene_tokens(
                input.overhead_scene_description, self.vocabulary)
            overhead_pos_embeddings = clevr_tokenizer.create_position_embeddings(
                input.overhead_scene_description, embedding_size=self.token_dim, min_scale=1e-6).to(self.device)
            overhead_scene_tokens = (
                self._overhead_vector_embedding(overhead_result['tokens'].to(self.device))
                + overhead_pos_embeddings
                + self._overhead_marker
            )
            overhead_scene_mask = overhead_result['mask'].to(self.device)
            overhead_tokens.append(overhead_scene_tokens)
            overhead_masks.append(overhead_scene_mask)
        if input.overhead_image is not None:
            overhead_image = self.image_preprocessing_transform(input.overhead_image)
            overhead_image_tokens = self._overhead_image_tokenizer(overhead_image)
            overhead_image_tokens = overhead_image_tokens + self._overhead_marker
            overhead_tokens.append(overhead_image_tokens)
            overhead_masks.append(torch.zeros(
                overhead_image_tokens.shape[0], overhead_image_tokens.shape[1], device=self.device))

        # tokenize ego information
        ego_tokens = []
        ego_masks = []
        if input.ego_scene_description is not None:
            ego_result = clevr_tokenizer.create_scene_tokens(
                input.ego_scene_description, self.vocabulary)
            ego_pos_embeddings = clevr_tokenizer.create_position_embeddings(
                input.ego_scene_description, embedding_size=self.token_dim, min_scale=1e-6).to(self.device)

            ego_scene_tokens = (
                self._ego_vector_embedding(ego_result['tokens'].to(self.device))
                + ego_pos_embeddings
                + self._ego_marker
            )
            ego_scene_mask = ego_result['mask'].to(self.device)
            ego_tokens.append(ego_scene_tokens)
            ego_masks.append(ego_scene_mask)
        if input.ego_image is not None:
            ego_image = self.image_preprocessing_transform(input.ego_image)
            ego_image_tokens = self._ego_image_tokenizer(ego_image)
            ego_image_tokens = ego_image_tokens + self._ego_marker
            ego_tokens.append(ego_image_tokens)
            ego_masks.append(torch.zeros(
                ego_image_tokens.shape[0], ego_image_tokens.shape[1], device=self.device))

        input_tokens = torch.cat(overhead_tokens + ego_tokens, dim=1)
        input_mask = torch.cat(overhead_masks + ego_masks, dim=1)

        embedded_tokens = self._encoder(
            input_tokens, src_key_padding_mask=input_mask, is_causal=False
        )

        out = {}
        match self._inference_method:
            case InferenceMethod.MEAN:
                out['mean'] = self.inference_mean(embedded_tokens, mask=input_mask)
                out['prediction'] = out["mean"]

            case InferenceMethod.HISTOGRAM:
                out['histogram'] = self.inference_histogram(
                    embedded_tokens,
                    mask=input_mask,
                    query_tokens=query_tokens,
                    query_mask=query_mask)

            case InferenceMethod.LEARNED_CORRESPONDENCE:
                assert input.ego_image is None and input.overhead_image is None, "Cannot do learned correspondences with image inputs"
                learned_correspondence, optimal_pose = self.inference_learned_correspondence(
                    embedded_tokens,
                    overhead_mask=overhead_masks[0],
                    ego_mask=ego_masks[0],
                    obj_in_overhead=positions_from_scene_descriptions(
                        input.overhead_scene_description).to(self.device),
                    obj_in_ego=positions_from_scene_descriptions(
                        input.ego_scene_description).to(self.device)
                )
                out["learned_correspondence"] = learned_correspondence
                if optimal_pose is not None:
                    out["prediction"] = optimal_pose

            case InferenceMethod.OPTIMIZED_POSE:
                assert input.ego_image is None and input.overhead_image is None, "Cannot do optimized pose with image inputs"
                learned_correspondence, optimal_pose = self.inference_learned_correspondence(
                    embedded_tokens,
                    overhead_mask=overhead_masks[0],
                    ego_mask=ego_masks[0],
                    obj_in_overhead=positions_from_scene_descriptions(
                        input.overhead_scene_description).to(self.device),
                    obj_in_ego=positions_from_scene_descriptions(
                        input.ego_scene_description).to(self.device)
                )
                out["learned_correspondence"] = learned_correspondence
                out["prediction"] = optimal_pose
            case _:
                raise NotImplementedError(
                    f"Unimplemented inference method: {self._inference_method}")

        return out
