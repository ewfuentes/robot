
import numpy as np

from experimental.beacon_dist.utils import KeypointDescriptorDtype


def get_test_data():
    return np.array(
        [
            (0, 1, 2, 3, 4, 5, 6, 7, np.ones((32,), dtype=np.int16) * 1, 1),
            (0, 8, 9, 10, 11, 12, 13, 14, np.ones((32,), dtype=np.int16) * 2, 1),
            (0, 15, 16, 17, 18, 19, 20, 21, np.ones((32,), dtype=np.int16) * 3, 2),
            (1, 22, 23, 24, 25, 26, 27, 28, np.ones((32,), dtype=np.int16) * 4, 1),
            (1, 29, 30, 31, 32, 33, 34, 35, np.ones((32,), dtype=np.int16) * 5, 3),
            (2, 36, 37, 38, 39, 40, 41, 42, np.ones((32,), dtype=np.int16) * 6, 8),
        ],
        dtype=KeypointDescriptorDtype,
    )
