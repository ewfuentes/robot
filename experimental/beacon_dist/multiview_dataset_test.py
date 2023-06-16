import unittest

import numpy as np

import experimental.beacon_dist.multiview_dataset as mvd
from experimental.beacon_dist.utils import KeypointDescriptorDtype, DESCRIPTOR_SIZE


def descriptor(i: int):
    return np.ones(DESCRIPTOR_SIZE, dtype=np.int16) * i


def get_test_dataset() -> list[dict[str, np.ndarray]]:
    keypoint_data_1 = np.array(
        [
            (0, 1, 2, 3, 4, 5, 6, 7, descriptor(1), (1, 0, 0, 0)),
            (0, 7, 8, 9, 10, 11, 12, 13, descriptor(2), (1, 0, 0, 0)),
            (0, 13, 14, 15, 16, 17, 18, 19, descriptor(3), (1, 0, 0, 0)),
            (1, 19, 20, 21, 22, 23, 24, 25, descriptor(4), (1, 0, 0, 0)),
            (1, 25, 26, 27, 28, 29, 30, 31, descriptor(5), (1, 0, 0, 0)),
            (1, 31, 32, 33, 34, 35, 36, 37, descriptor(6), (1, 0, 0, 0)),
            (2, 37, 38, 39, 40, 41, 42, 43, descriptor(7), (1, 0, 0, 0)),
            (2, 43, 44, 45, 46, 47, 48, 49, descriptor(8), (2, 0, 0, 0)),
            (2, 49, 50, 51, 52, 53, 54, 55, descriptor(9), (4, 0, 0, 0)),
        ],
        dtype=KeypointDescriptorDtype,
    )

    keypoint_data_2 = np.array(
        [
            (3, 55, 56, 57, 58, 59, 60, 61, descriptor(10), (1, 0, 0, 0)),
            (3, 61, 62, 63, 64, 65, 66, 67, descriptor(11), (2, 0, 0, 0)),
            (3, 67, 68, 69, 70, 71, 72, 73, descriptor(12), (4, 0, 0, 0)),
            (4, 73, 74, 75, 76, 77, 78, 79, descriptor(13), (8, 0, 0, 0)),
            (4, 79, 80, 81, 82, 83, 84, 85, descriptor(14), (8, 0, 0, 0)),
            (4, 85, 86, 87, 88, 89, 90, 91, descriptor(15), (8, 0, 0, 0)),
        ],
        dtype=KeypointDescriptorDtype,
    )

    image_info_1 = np.array(
        [
            (0, 0, np.zeros((3, 4))),
            (1, 0, np.zeros((3, 4))),
            (2, 0, np.zeros((3, 4))),
        ],
        dtype=mvd.ImageInfoDtype,
    )

    image_info_2 = np.array(
        [
            (3, 1, np.zeros((3, 4))),
            (4, 1, np.zeros((3, 4))),
        ],
        dtype=mvd.ImageInfoDtype,
    )

    object_list = ['obj_1', 'obj_2', 'obj_3']

    return [
        {
            "data": keypoint_data_1,
            "image_info": image_info_1,
            "object_list": object_list,
        },
        {
            "data": keypoint_data_2,
            "image_info": image_info_2,
            "object_list": object_list,
        },
    ]


class MultiviewDatasetTest(unittest.TestCase):
    def test_index_creation(self):
        # Setup
        dataset_inputs = mvd.DatasetInputs(
            file_paths=None, index_path=None, data_tables=get_test_dataset()
        )

        # Action
        dataset = mvd.MultiviewDataset(dataset_inputs)

        # Verification
        self.assertEqual(
            dataset._index.image_index[0],
            mvd.ImageIndexEntry(id=0, partition_idx=0, start_idx=0, end_idx=3),
        )
        self.assertEqual(
            dataset._index.image_index[1],
            mvd.ImageIndexEntry(id=1, partition_idx=0, start_idx=3, end_idx=6),
        )
        self.assertEqual(
            dataset._index.image_index[2],
            mvd.ImageIndexEntry(id=2, partition_idx=0, start_idx=6, end_idx=9),
        )
        self.assertEqual(
            dataset._index.image_index[3],
            mvd.ImageIndexEntry(id=3, partition_idx=1, start_idx=0, end_idx=3),
        )
        self.assertEqual(
            dataset._index.image_index[4],
            mvd.ImageIndexEntry(id=4, partition_idx=1, start_idx=3, end_idx=6),
        )

        self.assertEqual(
            dataset._index.scene_index[0],
            mvd.SceneIndexEntry(id=0, image_ids=[0, 1, 2]),
        )
        self.assertEqual(
            dataset._index.scene_index[1],
            mvd.SceneIndexEntry(id=1, image_ids=[3, 4]),
        )

        # an entry is a pair of a context and query points. Repeats are allowed
        # so there should be \sum_{scenes} len(scene)**2
        self.assertEqual(len(dataset), 3**2 + 2**2)

    def test_sample_retrival(self):
        # Setup
        dataset_inputs = mvd.DatasetInputs(
            file_paths=None, index_path=None, data_tables=get_test_dataset()
        )

        # Action
        dataset = mvd.MultiviewDataset(dataset_inputs)
        sample = dataset[10]

        # Verification
        # Fix me
        self.assertEqual(sample.query.image_id, 0.0)


if __name__ == "__main__":
    unittest.main()
