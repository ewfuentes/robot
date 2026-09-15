import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.analysis import correspondence_explorer
from experimental.overhead_matching.swag.evaluation.correspondence_matching import (
    RawCorrespondenceData,
)


class CorrespondenceExplorerTest(unittest.TestCase):

    def test_requires_raw_identity(self):
        raw = RawCorrespondenceData(
            cost_matrix=np.zeros((0, 0), dtype=np.float32),
            pano_id_to_lm_rows={},
            pano_lm_tags=[],
            osm_lm_indices=[],
            osm_lm_tags=[],
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "no source/alignment identity"):
                correspondence_explorer.validate_precomputed_data(
                    raw, object(), Path(temporary), "v1", None)


if __name__ == "__main__":
    unittest.main()
