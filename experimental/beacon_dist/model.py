import common.torch.torch as torch

from typing import NamedTuple

from experimental.beacon_dist import utils


class ConfigurationModelParams(NamedTuple):
    descriptor_size: int
    descriptor_embedding_size: int
    position_encoding_factor: float
    num_encoder_heads: int
    num_encoder_layers: int
    num_decoder_heads: int
    num_decoder_layers: int


def expand_descriptor(descriptor_tensor: torch.Tensor):
    BITS_PER_BYTE = 8
    shape = descriptor_tensor.shape
    new_shape = *shape[:-1], shape[-1] * BITS_PER_BYTE
    out = torch.zeros(new_shape, device=descriptor_tensor.device)
    for i in range(BITS_PER_BYTE):
        out[:, :, i::BITS_PER_BYTE] = torch.bitwise_and(descriptor_tensor, 1 << i) > 0
    return out


def encode_position(x: torch.Tensor, y: torch.Tensor, output_size: int, factor: float):
    assert output_size % 4 == 0, "Output size must be divisible by 4"
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    out = torch.zeros((*x.shape, output_size), dtype=torch.float32, device=x.device)
    for i in range(0, output_size, 4):
        out[:, :, i + 0] = torch.sin(x / (factor ** (i / output_size)))
        out[:, :, i + 1] = torch.cos(x / (factor ** (i / output_size)))
        out[:, :, i + 2] = torch.sin(y / (factor ** (i / output_size)))
        out[:, :, i + 3] = torch.cos(y / (factor ** (i / output_size)))
    return out


class ConfigurationModel(torch.nn.Module):
    def __init__(self, params: ConfigurationModelParams):
        super().__init__()

        # descriptor embedding
        self._descriptor_embedding = torch.nn.Linear(
            params.descriptor_size,
            params.descriptor_embedding_size,
            bias=False,
            dtype=torch.float32,
        )

        # encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=params.descriptor_embedding_size,
            nhead=params.num_encoder_heads,
            batch_first=True,
        )
        self._encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=params.num_encoder_layers
        )

        # decoder
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=params.descriptor_embedding_size,
            nhead=params.num_decoder_heads,
            batch_first=True,
        )
        self._decoder = torch.nn.TransformerDecoder(
            decoder_layer,
            num_layers=params.num_decoder_layers,
        )

        self._valid_classification_token = torch.nn.Parameter(
            data=torch.randn(1, 1, params.descriptor_embedding_size)
        )

        self._classifier = torch.nn.Linear(params.descriptor_embedding_size, 1)

        self._params = params

    def encode_descriptors(self, descriptors: torch.Tensor):
        expanded_descriptors = expand_descriptor(descriptors)
        return self._descriptor_embedding(expanded_descriptors)

    def position_encoding(self, x: torch.Tensor, y: torch.Tensor):
        return encode_position(
            x,
            y,
            self._params.descriptor_embedding_size,
            self._params.position_encoding_factor,
        )

    def forward(
        self, context_and_query: utils.KeypointPairs, configuration: torch.Tensor
    ):
        context_data, query_data = context_and_query
        # Configuration is a [batch, keypoint] binary tensor where we want the model to
        # compute the probability that the configuration is valid
        assert configuration.ndim == 2
        assert configuration.shape[0] == query_data.x.shape[0]
        batch_size = configuration.shape[0]

        # Encode the descriptors
        context_descriptors = self.encode_descriptors(
            context_data.descriptor
        ) + self.position_encoding(context_data.x, context_data.y)

        # Pass through transformer encoder layers
        # TODO mask out padding entries
        encoded_context = self._encoder(context_descriptors)

        # Encode the query
        query_descriptors = self.encode_descriptors(
            query_data.descriptor
        ) + self.position_encoding(query_data.x, query_data.y)

        classification_token = self._valid_classification_token.repeat(batch_size, 1, 1)

        query_descriptors_with_classification_token = torch.cat(
            [classification_token, query_descriptors], dim=1
        )

        expanded_query = torch.cat(
            [
                torch.ones(
                    batch_size, 1, dtype=torch.bool, device=configuration.device
                ),
                configuration,
            ],
            dim=1,
        )

        # add padding masks for the target and memory
        decoder_output = self._decoder(
            tgt=query_descriptors_with_classification_token,
            memory=encoded_context,
            tgt_key_padding_mask=torch.logical_not(expanded_query),
        )

        # Average pooling across the keypoints
        # TODO consider using attention based pooling instead
        average_output = torch.mean(decoder_output, dim=1)

        # Produce the log probability of this being a valid configuration
        return self._classifier(average_output)
