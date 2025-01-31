
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
            d_model=config.token_dim,
            nhead=config.num_encoder_heads,
            batch_first=True
        )
        self._encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers)

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=config.token_dim,
            nhead=config.num_decoder_heads,
        )
        self._decoder = torch.nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers)

        self._overhead_marker = torch.nn.Parameter(torch.randn(config.token_dim))
        self._ego_marker = torch.nn.Parameter(torch.randn(config.token_dim))

        self._ego_vector_embedding = torch.nn.Embedding(
            num_embeddings=config.vocabulary_size,
            embedding_dim=config.token_dim)

        self._overhead_vector_embedding = torch.nn.Embedding(
            num_embeddings=config.vocabulary_size,
            embedding_dim=config.token_dim)


    def forward(self,
                input: ClevrInputTokens,
                query_tokens: torch.Tensor,
                query_mask: torch.Tensor):

        overhead_tokens = self._overhead_vector_embedding(input.overhead_tokens) + input.overhead_position + self._overhead_marker
        ego_tokens = self._ego_vector_embedding(input.ego_tokens) + input.ego_position + self._ego_marker

        input_tokens = torch.cat([overhead_tokens, ego_tokens], dim=1)
        input_mask = torch.cat([input.overhead_mask, input.ego_mask], dim=1)

        embedded_tokens = self._encoder(input_tokens, input_mask)

        output_tokens = self._decoder(
            tgt=query_tokens,
            tgt_mask=query_mask,
            memory=embedded_tokens,
            memory_mask=input_mask)

        
