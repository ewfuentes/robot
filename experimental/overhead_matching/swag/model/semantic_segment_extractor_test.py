
import unittest

import matplotlib.pyplot as plt

from pathlib import Path
import supervision as sv
import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.model.semantic_segment_extractor as sse
import numpy as np
import tqdm


class SAM2SegmentExtractorTest(unittest.TestCase):
    def test_sample_image(self):
        # Setup
        BATCH_SIZE = 30
        CLIP_FEATURE_DIM = 512
        dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet"),
            vd.VigorDatasetConfig(panorama_neighbor_radius=1e-6))
        dataloader = vd.get_dataloader(dataset.get_pano_view(), batch_size=BATCH_SIZE)

        model = sse.SemanticSegmentExtractor(
            sse.SemanticSegmentExtractorConfig()
        )

        # Action
        batch = next(iter(dataloader))
        positions, tokens, mask = model(sse.ModelInput(
            image=batch.panorama, metadata=batch.panorama_metadata))

        self.assertEqual(positions.shape[0], BATCH_SIZE)
        self.assertEqual(positions.shape[2], 2)
        self.assertEqual(tokens.shape[0], BATCH_SIZE)
        self.assertEqual(tokens.shape[2], CLIP_FEATURE_DIM)
        self.assertEqual(mask.shape[0], BATCH_SIZE)
        self.assertEqual(mask.shape[1], tokens.shape[1])
        self.assertEqual(mask.shape[1], positions.shape[1])


if __name__ == "__main__":
    unittest.main()
