
import unittest

import matplotlib.pyplot as plt

from pathlib import Path
import supervision as sv
import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.model.semantic_segment_extractor as sse
import numpy as np
import tqdm


class SemanticSegmentExtractorTest(unittest.TestCase):
    def test_sample_image(self):
        # Setup
        BATCH_SIZE = 30
        CLIP_FEATURE_DIM = 512
        dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet"),
            vd.VigorDatasetConfig(panorama_neighbor_radius=1e-6))
        dataloader = vd.get_dataloader(dataset.get_pano_view(), batch_size=BATCH_SIZE)

        model = sse.SemanticSegmentExtractor(
            sse.SemanticSegmentExtractorConfig(points_per_batch=32)
        )

        # Action
        batch = next(iter(dataloader))
        out = model(sse.ModelInput(
            image=batch.panorama, metadata=batch.panorama_metadata))

        self.assertEqual(out.positions.shape[0], BATCH_SIZE)
        self.assertEqual(out.positions.shape[2], 2)
        self.assertEqual(out.features.shape[0], BATCH_SIZE)
        self.assertEqual(out.features.shape[2], CLIP_FEATURE_DIM)
        self.assertEqual(out.mask.shape[0], BATCH_SIZE)
        self.assertEqual(out.mask.shape[1], out.features.shape[1])
        self.assertEqual(out.mask.shape[1], out.positions.shape[1])


if __name__ == "__main__":
    unittest.main()
