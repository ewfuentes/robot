
import unittest

import numpy as np

from experimental.learn_descriptors.translation_prior_python import TranslationPrior


class TranslationPriorPythonTest(unittest.TestCase):
    def test_construct_and_access(self):
        # Action
        translation = np.array([1.0,1.0,1.0])
        covariance = np.array([0.94, 0.01, 0.05, 0.01, 0.94, 0.05, 0.05, 0.05, 0.9]).reshape((3,3))
        translation_prior = TranslationPrior(translation, covariance)

        # Verification
        self.assertEqual()