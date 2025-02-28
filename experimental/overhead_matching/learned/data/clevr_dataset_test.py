import unittest

import common.torch.load_torch_deps
import torch
from experimental.overhead_matching.learned.data import clevr_dataset
from pathlib import Path


class ClevrDatasetTest(unittest.TestCase):
    def test_dataloader(self):
        # Setup
        BATCH_SIZE = 4
        dataset = clevr_dataset.ClevrDataset(
            Path("external/clevr_test_set/clevr_test_set")
        )
        loader = clevr_dataset.get_dataloader(dataset, batch_size=BATCH_SIZE)

        # Action
        batch = next(iter(loader))

        # Verification
        self.assertEqual(len(batch.scene_description["objects"]), BATCH_SIZE)
        self.assertEqual(batch.overhead_image, None)
        self.assertEqual(batch.ego_image, None)


    def test_overhead_dataloader(self):

        BATCH_SIZE = 4
        dataset = clevr_dataset.ClevrDataset(
            Path("external/clevr_test_set/clevr_test_set"),
            load_overhead=True
        )
        loader = clevr_dataset.get_dataloader(dataset, batch_size=BATCH_SIZE)

        # Action
        batch = next(iter(loader))

        self.assertEqual(len(batch.scene_description["objects"]), BATCH_SIZE)
        self.assertTrue(torch.is_tensor(batch.overhead_image))
        self.assertEqual(batch.overhead_image.shape, (BATCH_SIZE, 3, 240, 320))
        self.assertEqual(batch.overhead_image.dtype, torch.float32)
        self.assertEqual(batch.ego_image, None)

    def test_ego_dataloader(self):

        BATCH_SIZE = 8
        dataset = clevr_dataset.ClevrDataset(
            Path("external/clevr_test_set/clevr_test_set"),
            load_ego_images=True
        )
        loader = clevr_dataset.get_dataloader(dataset, batch_size=BATCH_SIZE)

        # Action
        batch = next(iter(loader))

        self.assertEqual(len(batch.scene_description["objects"]), BATCH_SIZE)
        self.assertTrue(torch.is_tensor(batch.ego_image))
        self.assertEqual(batch.ego_image.shape, (BATCH_SIZE, 3, 240, 320))
        self.assertEqual(batch.ego_image.dtype, torch.float32)
        self.assertEqual(batch.overhead_image, None)

    def test_ego_not_overhead(self):

        BATCH_SIZE = 8
        dataset = clevr_dataset.ClevrDataset(
            Path("external/clevr_test_set/clevr_test_set"),
            load_ego_images=True,
            load_overhead=True

        )
        loader = clevr_dataset.get_dataloader(dataset, batch_size=BATCH_SIZE)

        # Action
        batch = next(iter(loader))

        self.assertEqual(len(batch.scene_description["objects"]), BATCH_SIZE)
        self.assertTrue(torch.is_tensor(batch.overhead_image))
        self.assertTrue(torch.is_tensor(batch.ego_image))
        self.assertFalse(torch.allclose(batch.ego_image, batch.overhead_image))



if __name__ == "__main__":
    unittest.main()
