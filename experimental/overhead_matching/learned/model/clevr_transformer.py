import common.torch.torch as torch

from dataclasses import dataclass


@dataclass
class ClevrTransformerConfig:
    token_dim: int
    vocabulary_size: int
    num_encoder_heads: int
    num_encoder_layers: int
    num_decoder_heads: int
    num_decoder_layers: int
    output_dim: int
    predict_gaussian: bool


@dataclass
class ClevrInputTokens:
    overhead_tokens: torch.Tensor
    overhead_position: torch.Tensor
    overhead_position_embeddings: torch.Tensor
    overhead_mask: torch.Tensor
    ego_tokens: torch.Tensor
    ego_position: torch.Tensor
    ego_position_embeddings: torch.Tensor
    ego_mask: torch.Tensor


class ClevrTransformer(torch.nn.Module):
    def __init__(self, config: ClevrTransformerConfig):
        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.num_encoder_heads,
            batch_first=True,
            dropout=0.0,
        )
        self._encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=config.token_dim,
            nhead=config.num_decoder_heads,
            batch_first=True,
            dropout=0.0,
        )
        self._decoder = torch.nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers
        )

        self._overhead_marker = torch.nn.Parameter(torch.randn(config.token_dim))
        self._ego_marker = torch.nn.Parameter(torch.randn(config.token_dim))

        self._predict_gaussian = None
        if config.predict_gaussian:
            self._predict_gaussian = torch.nn.Parameter(torch.randn(config.token_dim))

        self._ego_vector_embedding = torch.nn.Embedding(
            num_embeddings=config.vocabulary_size, embedding_dim=config.token_dim
        )

        self._overhead_vector_embedding = torch.nn.Embedding(
            num_embeddings=config.vocabulary_size, embedding_dim=config.token_dim
        )

        self._output_layer = torch.nn.Linear(config.token_dim, config.output_dim)

        self._correspondence_no_match_token = torch.nn.Parameter(torch.randn(config.token_dim))

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

        softmax_mask = torch.zeros((batch_size, num_overhead_tokens, num_ego_tokens+1), device=embedded_tokens.device)
        softmax_mask[..., :-1] = torch.logical_or(softmax_mask[..., :-1], valid_overhead_mask)
        softmax_mask[..., :-1] = torch.logical_or(softmax_mask[..., :-1], valid_ego_mask)
        softmax_mask[softmax_mask == True] = -torch.inf

        return torch.softmax(attention_logits + softmax_mask, dim=-1)

    def forward(
        self,
        input: ClevrInputTokens,
        query_tokens: None | torch.Tensor,
        query_mask: None | torch.Tensor,
    ):

        overhead_tokens = (
            self._overhead_vector_embedding(input.overhead_tokens)
            + input.overhead_position_embeddings
            + self._overhead_marker
        )
        ego_tokens = (
            self._ego_vector_embedding(input.ego_tokens)
            + input.ego_position_embeddings
            + self._ego_marker
        )

        input_tokens = torch.cat([overhead_tokens, ego_tokens], dim=1)
        input_mask = torch.cat([input.overhead_mask, input.ego_mask], dim=1)

        batch_size = input_tokens.shape[0]

        embedded_tokens = self._encoder(
            input_tokens, src_key_padding_mask=input_mask, is_causal=False
        )

        learned_correspondence = self.compute_learned_correspondence(
                embedded_tokens, input.overhead_mask, input.ego_mask)

        if self._predict_gaussian is not None:
            query_tokens = self._predict_gaussian.expand(batch_size, 1, -1)
            query_mask = None

        output_tokens = self._decoder(
            tgt=query_tokens,
            tgt_key_padding_mask=query_mask,
            memory=embedded_tokens,
            memory_key_padding_mask=input_mask,
            tgt_is_causal=False,
            memory_is_causal=False,
        )

        output = self._output_layer(output_tokens)
        if self._predict_gaussian is not None:
            output = output.squeeze(1)

        return {
            'decoder_output': output,
            'learned_correspondence': learned_correspondence
        }
