import common.torch as torch

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
    overhead_mask: torch.Tensor
    ego_tokens: torch.Tensor
    ego_position: torch.Tensor
    ego_mask: torch.Tensor


class ClevrTransformer(torch.nn.Module):
    def __init__(self, config: ClevrTransformerConfig):
        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.token_dim, nhead=config.num_encoder_heads, batch_first=True, dropout=0.0
        )
        self._encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=config.token_dim, nhead=config.num_decoder_heads, batch_first=True, dropout = 0.0
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

    def forward(
        self,
        input: ClevrInputTokens,
        query_tokens: None | torch.Tensor,
        query_mask: None | torch.Tensor,
    ):

        overhead_tokens = (
            self._overhead_vector_embedding(input.overhead_tokens)
            + input.overhead_position
            + self._overhead_marker
        )
        ego_tokens = (
            self._ego_vector_embedding(input.ego_tokens)
            + input.ego_position
            + self._ego_marker
        )

        input_tokens = torch.cat([overhead_tokens, ego_tokens], dim=1)
        input_mask = torch.cat([input.overhead_mask, input.ego_mask], dim=1)

        batch_size = input_tokens.shape[0]

        embedded_tokens = self._encoder(
            input_tokens, src_key_padding_mask=input_mask, is_causal=False
        )

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
            return output.squeeze(1)

        return output
