import unittest

import numpy as np
import torch

from experimental.beacon_dist.test_helpers import get_test_data
from experimental.beacon_dist.utils import (
    Dataset,
    reconstruction_loss,
    sample_keypoints,
    batchify,
    valid_configuration_loss,
)


class DatasetTest(unittest.TestCase):
    def test_dataset_from_filename(self):
        # Setup
        test_data = get_test_data()
        # Action
        dataset = Dataset(data=test_data)

        # Verification
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[1].descriptor[0, 0], 4)


class UtilsTest(unittest.TestCase):
    def test_sample_keypoints_subsample(self):
        # Setup
        gen = torch.manual_seed(0)
        test_data = get_test_data()
        dataset = Dataset(data=test_data)

        # Action
        subsampled = sample_keypoints(dataset[0], num_keypoints_to_sample=2, gen=gen)

        # Verification
        # There should be two keypoints even though the original sample had 3 points
        self.assertEqual(subsampled.x.shape[0], 2)

    def test_batchify(self):
        # Setup
        test_data = get_test_data()
        dataset = Dataset(data=test_data)
        batch = dataset._data

        # Action
        batched = batchify(batch)

        # Verification
        # Check that one of the fields is set appropriately
        # There are three elements in the batch and the largest one has three key points
        self.assertEqual(batched.x.shape, (3, 3))

        # The second element has two keypoints, so the last element should be float32.min
        self.assertNotEqual(batched.x[1, 0], torch.finfo(torch.float32).min)
        self.assertNotEqual(batched.x[1, 1], torch.finfo(torch.float32).min)
        self.assertEqual(batched.x[1, 2], torch.finfo(torch.float32).min)

        # The third element has one keypoint, so the last two elements should be float32.min
        self.assertNotEqual(batched.x[2, 0], torch.finfo(torch.float32).min)
        self.assertEqual(batched.x[2, 1], torch.finfo(torch.float32).min)
        self.assertEqual(batched.x[2, 2], torch.finfo(torch.float32).min)

    def test_reconstructor_loss(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch_in = batchify(dataset._data)
        batch_out = batch_in._replace(x=batch_in.x + 0.1)
        expected_loss = torch.mean((batch_out.x - batch_in.x) ** 2)

        # Action
        loss = reconstruction_loss(batch_in, batch_out)

        # Verification
        self.assertEqual(loss, expected_loss)

    def test_valid_configuration_loss_valid_queries_are_likely(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch = batchify(dataset._data)
        query = torch.tensor([[1, 1, 1],
                              [1, 1, 0],
                              [1, 0, 0]])
        model_output = torch.tensor([[100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch, query, model_output)

        # Verification
        self.assertAlmostEqual(loss, 0.0, 1e-6)

    def test_valid_configuration_loss_invalid_queries_are_unlikely(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch = batchify(dataset._data)
        query = torch.tensor([[0, 1, 1],
                              [1, 1, 0],
                              [1, 0, 0]])
        model_output = torch.tensor([[-100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch, query, model_output)

        # Verification
        self.assertAlmostEqual(loss, 0.0, 1e-6)

    def test_valid_configuration_loss_common_keypoints_are_ignored(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch = batchify(dataset._data)
        query = torch.tensor([[1, 1, 1],
                              [1, 0, 0],
                              [1, 0, 0]])
        model_output = torch.tensor([[100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch, query, model_output)

        # Verification
        self.assertAlmostEqual(loss, 0.0, 1e-6)

    def test_valid_configuration_loss_bad_prediction_yields_nonzero_loss(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch = batchify(dataset._data)
        query = torch.tensor([[1, 1, 1],
                              [1, 1, 0],
                              [1, 0, 0]])
        model_output = torch.tensor([[-100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch, query, model_output)

        # Verification
        self.assertGreater(loss, 0.0)

    def test_valid_configuration_loss_missing_exclusive_keypoints_are_valid(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch = batchify(dataset._data)
        query = torch.tensor([[0, 0, 1],
                              [1, 1, 0],
                              [1, 0, 0]])
        model_output = torch.tensor([[100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch, query, model_output)

        # Verification
        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
