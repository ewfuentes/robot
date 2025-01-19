
import unittest

from pathlib import Path
import numpy as np

import experimental.overhead_matching.cvusa_evaluation_pipeline as cep

class CVUSAEvaluationPipelineTest(unittest.TestCase):
    def test_happy_case(self):
        # Setup
        dataset_path = Path("external/cvusa_minisubset")

        def test_method(*, ego, overhead):
            ego_height, ego_width, _ = ego.shape
            ego_aspect = ego_width / ego_height
            self.assertGreater(ego_aspect, 2.0)
            return (0.0, 0.0)

        # Action
        result = cep.cvusa_evaluation(dataset_path, test_method)

        # Verification
        expected_mae = 0.5 * 2 ** 0.5
        self.assertAlmostEqual(result.mean_absolute_error, expected_mae, places=5)




if __name__ == "__main__":
    unittest.main()
