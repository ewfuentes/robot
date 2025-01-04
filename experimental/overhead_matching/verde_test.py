import unittest

import os
import common.testing.is_test_python as itp

if itp.is_test():
    os.environ['HF_HOME'] = os.environ['TEST_TMPDIR'] + "/huggingface"

from PIL import Image
from pathlib import Path
import numpy as np

from experimental.overhead_matching import verde
from experimental.overhead_matching import grounding_sam as gs

GROUNDING_DINO_MODEL_ID = 'IDEA-Research/grounding-dino-base'

class VerdeTest(unittest.TestCase):
    def test_happy_case(self):
        model = gs.GroundingSam(GROUNDING_DINO_MODEL_ID)
        dataset_path = Path("external/cvusa_minisubset")
        overhead_path = dataset_path / "bingmap/20/0000012.jpg"
        ego_path = dataset_path / "streetview/panos/0000012.jpg"

        self.assertTrue(overhead_path.exists())
        self.assertTrue(ego_path.exists())

        inputs = verde.OverheadMatchingInput(
            overhead=np.asarray(Image.open(overhead_path).convert("RGB")),
            ego=np.asarray(Image.open(ego_path).convert("RGB")),
        )

        verde.estimate_overhead_transform(inputs, model)

    def test_happy_case_2(self):
        model = gs.GroundingSam(GROUNDING_DINO_MODEL_ID)
        dataset_path = Path("external/cvusa_minisubset")
        overhead_path = dataset_path / "bingmap/20/0000011.jpg"
        ego_path = dataset_path / "streetview/panos/0000011.jpg"

        self.assertTrue(overhead_path.exists())
        self.assertTrue(ego_path.exists())

        inputs = verde.OverheadMatchingInput(
            overhead=np.asarray(Image.open(overhead_path).convert("RGB")),
            ego=np.asarray(Image.open(ego_path).convert("RGB")),
        )

        verde.estimate_overhead_transform(inputs, model)


if __name__ == "__main__":
    unittest.main()
