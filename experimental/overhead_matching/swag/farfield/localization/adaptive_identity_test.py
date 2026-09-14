import tempfile
import unittest
from pathlib import Path
import msgspec
from experimental.overhead_matching.swag.farfield.localization import adaptive_identity as ai, structs
import torch


class AdaptiveIdentityTest(unittest.TestCase):
    def test_heading_uncertainty_does_not_hide_position_lock(self):
        prior = torch.zeros((4, 1, 3))
        prior[:, 0, 1] = 0.25
        east = torch.tensor([0., 1000., 2000.])
        north = torch.zeros(3)
        self.assertEqual(ai.spatial_mode_mass(prior, east, north, 500), 1.)

    def test_two_sharp_separated_modes_remain_uncertain(self):
        prior = torch.tensor([[[0.5, 0., 0.5]]])
        east = torch.tensor([0., 1000., 2000.])
        north = torch.zeros(3)
        self.assertEqual(ai.spatial_mode_mass(prior, east, north, 500), .5)
        # The next decision must see the updated belief, not a saved prefix's.
        posterior = prior * torch.tensor([[[19., 1., 1.]]])
        self.assertAlmostEqual(ai.spatial_mode_mass(posterior, east, north, 500), .95)

    def test_boundary_and_normalization(self):
        prior = torch.tensor([[[3., 1., 1.]]])
        east = torch.tensor([0., 500., 501.])
        self.assertAlmostEqual(ai.spatial_mode_mass(prior, east, torch.zeros(3), 500), .8)

    def test_alternatives_require_exact_identity_and_valid_scores(self):
        table = structs.CompatibilityTable(tracklet_id='track#T1', matcher_version='a',
            entries=[], default_log_lr=-12., clip_lo=-12., clip_hi=4., status='fast')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tables.json'
            path.write_bytes(msgspec.json.encode([msgspec.structs.replace(table, matcher_version='b')]))
            self.assertEqual(ai.load_alternatives(path, {table.tracklet_id: table}), {})
            path.write_bytes(msgspec.json.encode([table, table]))
            with self.assertRaises(ValueError):
                ai.load_alternatives(path, {table.tracklet_id: table})
            path.write_bytes(msgspec.json.encode([msgspec.structs.replace(table, tracklet_id='other#T1')]))
            with self.assertRaises(ValueError):
                ai.load_alternatives(path, {table.tracklet_id: table})
            entry = structs.CompatibilityEntry(landmark_id='map:a', log_lr=0.)
            path.write_bytes(msgspec.json.encode([msgspec.structs.replace(table, entries=[entry, entry])]))
            with self.assertRaises(ValueError):
                ai.load_alternatives(path, {table.tracklet_id: table})


if __name__ == '__main__':
    unittest.main()
