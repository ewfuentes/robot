import unittest

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    panorama_score,
)


def _random_db(n_loc: int = 40, n_theta: int = 12, dim: int = 32,
               seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    db = torch.randn(n_loc, n_theta, dim, generator=generator)
    return torch.nn.functional.normalize(db, dim=-1)


class JointScoreTest(unittest.TestCase):
    def test_recovers_location_and_shift(self):
        db = _random_db()
        true_loc, true_shift = 17, 5
        n_theta = db.shape[1]
        query = db[true_loc, (torch.arange(n_theta) + true_shift) % n_theta]

        result = panorama_score.joint_scores(query, db)
        _, loc_idx, shift_idx = result.top_k(1)
        self.assertEqual(int(loc_idx[0]), true_loc)
        self.assertEqual(int(shift_idx[0]), true_shift)
        self.assertAlmostEqual(
            result.heading_cw_deg[int(shift_idx[0])], true_shift * 30.0)
        self.assertAlmostEqual(
            float(result.scores[true_loc, true_shift]), 1.0, places=5)

    def test_invalid_crops_are_excluded(self):
        db = _random_db(seed=1)
        true_loc, true_shift = 3, 8
        n_theta = db.shape[1]
        query = db[true_loc,
                   (torch.arange(n_theta) + true_shift) % n_theta].clone()
        valid = torch.ones(n_theta, dtype=torch.bool)
        query[4] = torch.nn.functional.normalize(
            torch.randn(db.shape[2], generator=torch.Generator()
                        .manual_seed(9)), dim=0)
        valid[4] = False

        result = panorama_score.joint_scores(query, db, valid_crops=valid)
        _, loc_idx, shift_idx = result.top_k(1)
        self.assertEqual(int(loc_idx[0]), true_loc)
        self.assertEqual(int(shift_idx[0]), true_shift)
        self.assertAlmostEqual(
            float(result.scores[true_loc, true_shift]), 1.0, places=5)

    def test_all_invalid_raises(self):
        db = _random_db()
        query = db[0]
        with self.assertRaises(ValueError):
            panorama_score.joint_scores(
                query, db, valid_crops=torch.zeros(12, dtype=torch.bool))

    def test_mismatched_rings_raise(self):
        db = _random_db(n_theta=12)
        query = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
        with self.assertRaises(ValueError):
            panorama_score.joint_scores(query, db)

    def test_scores_shape(self):
        db = _random_db(n_loc=7, n_theta=12)
        result = panorama_score.joint_scores(db[0], db)
        self.assertEqual(tuple(result.scores.shape), (7, 12))
        self.assertEqual(len(result.heading_cw_deg), 12)


if __name__ == "__main__":
    unittest.main()
