import unittest

import numpy as np
import torch

from experimental.beacon_dist.utils import (
    Dataset,
    KeypointDescriptorDtype,
    sample_keypoints,
    batchify,
)


def get_test_data():
    return np.array(
        [
            (0, 1, 2, 3, 4, 5, 6, 7, np.ones((32,), dtype=np.int16) * 1),
            (0, 8, 9, 10, 11, 12, 13, 14, np.ones((32,), dtype=np.int16) * 2),
            (0, 15, 16, 17, 18, 19, 20, 21, np.ones((32,), dtype=np.int16) * 3),
            (1, 22, 23, 24, 25, 26, 27, 28, np.ones((32,), dtype=np.int16) * 4),
            (1, 29, 30, 31, 32, 33, 34, 35, np.ones((32,), dtype=np.int16) * 5),
            (2, 36, 37, 38, 39, 40, 41, 42, np.ones((32,), dtype=np.int16) * 6),
        ],
        dtype=KeypointDescriptorDtype,
    )


class DatasetTest(unittest.TestCase):
    def test_dataset_from_filename(self):
        # Setup
        test_data = get_test_data()
        # Action
        dataset = Dataset(data=test_data)

        # Verification
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[1].descriptor[0, 0, 0], 4)


class UtilsTest(unittest.TestCase):
    def test_sample_keypoints_subsample(self):
        # Setup
        gen = torch.manual_seed(0)
        test_data = get_test_data()
        dataset = Dataset(data=test_data)

        # Action
        subsampled = sample_keypoints(dataset[0], num_keypoints_to_sample=2, gen=gen)

        # Verification
        # The batch dimension should be one
        self.assertEqual(subsampled.image_id.shape[0], 1)
        # There should be two keypoints even though the original sample had 3 points
        self.assertEqual(subsampled.x.shape[1], 2)

    def test_batchify(self):
        # Setup
        gen = torch.manual_seed(0)
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


if __name__ == "__main__":
    unittest.main()
