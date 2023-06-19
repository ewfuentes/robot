import torch
import unittest

from experimental.beacon_dist import model
from experimental.beacon_dist.test_helpers import get_test_data
from experimental.beacon_dist.utils import batchify_pair, KeypointBatch, KeypointPairs
from experimental.beacon_dist.multiview_dataset import MultiviewDataset


def create_batch_from_dataset(dataset: MultiviewDataset) -> KeypointPairs:
    samples = list(dataset)
    fields = {}
    for field in KeypointBatch._fields:
        fields[field] = torch.nested.nested_tensor([getattr(x.context, field) for x in samples])
    single = KeypointBatch(**fields)
    return KeypointPairs(single, single)


class ModelTest(unittest.TestCase):
    def test_expand_descriptor(self):
        # Setup
        tensor_in = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=torch.int16)
        # fmt: off
        expected_result = torch.tensor([[
            [1, 0, 0, 0, 0, 0, 0, 0,  # 1
                0, 1, 0, 0, 0, 0, 0, 0,  # 2
                1, 1, 0, 0, 0, 0, 0, 0,  # 3
                0, 0, 1, 0, 0, 0, 0, 0,  # 4
                ],
            [1, 0, 1, 0, 0, 0, 0, 0,  # 5
                0, 1, 1, 0, 0, 0, 0, 0,  # 6
                1, 1, 1, 0, 0, 0, 0, 0,  # 7
                0, 0, 0, 1, 0, 0, 0, 0,  # 8
            ]]])
        # fmt: on

        # Action
        result = model.expand_descriptor(tensor_in)

        # Verification
        # We use equality here since we're working with an integer tensor
        self.assertEqual(torch.linalg.matrix_norm(result - expected_result), 0)

    def test_position_encoding(self):
        # Setup
        x = torch.tensor([[1, 2, 3, 4]])
        y = torch.tensor([[10, 9, 8, 4]])
        OUTPUT_SIZE = 16
        FACTOR = 10000

        # Action
        position_embedding = model.encode_position(x, y, OUTPUT_SIZE, FACTOR)

        # Verification
        self.assertEqual(position_embedding.shape, (*x.shape, OUTPUT_SIZE))

    def test_run_model(self):
        # Setup
        dataset = MultiviewDataset.from_single_view(data=get_test_data())
        batch = batchify_pair(create_batch_from_dataset(dataset))
        config = torch.Tensor(
            [[True, True, True], [True, True, False], [True, False, False]]
        ).to(torch.bool)
        m = model.ConfigurationModel(
            model.ConfigurationModelParams(
                descriptor_size=256,
                descriptor_embedding_size=32,
                position_encoding_factor=100000,
                num_encoder_heads=4,
                num_encoder_layers=2,
                num_decoder_heads=4,
                num_decoder_layers=2,
            )
        )

        # Action
        output = m(context_and_query=batch, configuration=config)

        # Verification
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
