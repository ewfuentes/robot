import unittest

import common.torch as torch
from experimental.overhead_matching.learned.data import clevr_dataset
from pathlib import Path

class ClevrDatasetTest(unittest.TestCase):
    def test_dataloader(self):
        # Setup
        BATCH_SIZE = 4
        dataset = clevr_dataset.ClevrDataset(
            Path("external/clevr_test_set/clevr_test_set"))
        loader = clevr_dataset.get_dataloader(dataset, batch_size=BATCH_SIZE)

        # Action
        batch = next(iter(loader))

        # Verification
        self.assertEqual(len(batch['objects']), BATCH_SIZE)

if __name__ == "__main__":
    unittest.main()
