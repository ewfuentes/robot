import torch
import unittest

from experimental.beacon_dist import model
from experimental.beacon_dist.test_helpers import get_test_data
from experimental.beacon_dist.utils import Dataset, batchify


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
        x = torch.tensor([[[1, 2, 3, 4]]])
        y = torch.tensor([[[10, 9, 8, 4]]])
        OUTPUT_SIZE = 16
        FACTOR = 10000

        # Action
        position_embedding = model.encode_position(x, y, OUTPUT_SIZE, FACTOR)

        # Verification
        print(position_embedding.shape)
        print(position_embedding)

    def test_run_model(self):
        # Setup
        batch = batchify(Dataset(data=get_test_data())._data)
        m = model.Reconstructor(model.ReconstructorParams(
            descriptor_size=256,
            descriptor_embedding_size=16,
            position_encoding_factor=100000
        ))

        # Action
        output = m(batch)

        # Verification
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
