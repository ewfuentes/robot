
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
        dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet"),
            vd.VigorDatasetConfig(panorama_neighbor_radius=1e-6))

        model = sse.SemanticSegmentExtractor(
            sse.SemanticSegmentExtractorConfig()
        )

        # Action
        for i in tqdm.tqdm(range(20)):
            item = dataset.get_pano_view()[i]

            image = item.panorama.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            result = model(image)
            # detections = sv.Detections.from_sam(sam_result=result)

            # annotated = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image.copy(), detections)
            # plt.figure()
            # plt.subplot(211)
            # plt.imshow(image)
            # plt.subplot(212)
            # plt.imshow(annotated)

            # plt.show(block=True)


if __name__ == "__main__":
    unittest.main()
