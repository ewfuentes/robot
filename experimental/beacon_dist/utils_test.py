
import unittest

import numpy as np

from experimental.beacon_dist.utils import Dataset, KeypointDescriptorDtype

class DatasetTest(unittest.TestCase):
    def test_dataset_from_filename(self):
        # Setup
        test_data = np.array([
            (0, 1, 2, 3, 4, 5, 6, 7, np.ones((32,), dtype=np.uint8) * 1),
            (0, 8, 9, 10, 11, 12, 13, 14, np.ones((32,), dtype=np.uint8) * 2),
            (0, 15, 16, 17, 18, 19, 20, 21, np.ones((32,), dtype=np.uint8) * 3),
            (1, 22, 23, 24, 25, 26, 27, 28, np.ones((32,), dtype=np.uint8) * 4),
            (1, 29, 30, 31, 32, 33, 34, 35, np.ones((32,), dtype=np.uint8) * 5),
            (2, 36, 37, 38, 39, 40, 41, 42, np.ones((32,), dtype=np.uint8) * 6),
        ], dtype=KeypointDescriptorDtype)

        # Action
        dataset = Dataset(data=test_data)

        # Verification
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[1].descriptor[0, 0], 4)


if __name__ == "__main__":
    unittest.main()

