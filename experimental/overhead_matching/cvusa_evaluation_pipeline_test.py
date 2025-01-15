
import unittest

from pathlib import Path
import numpy as np

import experimental.overhead_matching.cvusa_evaluation_pipeline as cep

def test_method(ego: np.ndarray, overhead: np.ndarray):
    return (0.5, 0.5)

class CVUSAEvaluationPipelineTest(unittest.TestCase):
    def test_happy_case(self):
        # Setup
        dataset_path = Path("external/cvusa_minisubset")
        self.assertTrue(dataset_path.exists())

        # Action

        # Verification




if __name__ == "__main__":
    unittest.main()
