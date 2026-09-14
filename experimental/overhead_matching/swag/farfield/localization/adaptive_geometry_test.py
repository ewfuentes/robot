import json
from pathlib import Path
import tempfile
import unittest
import torch
from experimental.overhead_matching.swag.farfield.localization import adaptive_geometry as ag


class AdaptiveGeometryTest(unittest.TestCase):
    def setUp(self):
        self.prior = torch.tensor([[[.95, .03, .02]]])
        self.point = torch.tensor([[[1., 100., 1.]]])
        self.east = torch.tensor([0., 1000., 2000.])
        self.north = torch.zeros(3)
        self.calls = 0

    def choose(self, footprint, **overrides):
        def compute():
            self.calls += 1
            return footprint
        kwargs = dict(large_extended=True, mode_mass=.95, radius_m=500., lock_mass=.9)
        kwargs.update(overrides)
        return ag.choose_factor(self.prior, self.point, compute, self.east, self.north,
                                lambda x:x/x.sum(), **kwargs)

    def test_disagreement_retains_nearby_footprint(self):
        original = self.prior.clone()
        footprint = torch.ones_like(self.point)
        chosen, decision = self.choose(footprint)
        self.assertIs(chosen, footprint)
        self.assertTrue(decision['extent_enabled'])
        self.assertEqual(decision['point_map_jump_m'], 1000.)
        self.assertTrue(torch.equal(self.prior, original))

    def test_agreeing_distant_correction_is_kept(self):
        chosen, decision = self.choose(self.point)
        self.assertIs(chosen, self.point)
        self.assertFalse(decision['extent_enabled'])
        self.assertEqual(self.calls, 1)

    def test_farther_footprint_is_not_selected(self):
        chosen, decision = self.choose(torch.tensor([[[1., 1., 100.]]]))
        self.assertIs(chosen, self.point)
        self.assertFalse(decision['extent_enabled'])

    def test_diffuse_small_and_boundary_updates_do_not_compute_footprint(self):
        for override in [dict(mode_mass=.899),dict(large_extended=False),dict(radius_m=1000.)]:
            chosen, _ = self.choose(torch.ones_like(self.point), **override)
            self.assertIs(chosen, self.point)
        self.assertEqual(self.calls, 0)

    def test_extent_readiness_and_future_rows(self):
        rows = {'T1':dict(available_keyframe=8,large_extended=True),
                'T2':dict(available_keyframe=20,large_extended=False)}
        with self.assertRaises(ValueError):
            ag.large_at_release(rows,'T1',7)
        self.assertTrue(ag.large_at_release(rows,'T1',8))
        rows['T2']['large_extended'] = True
        self.assertTrue(ag.large_at_release(rows,'T1',8))

    def test_loader_rejects_duplicate_or_missing_track_ids(self):
        row = dict(tracklet_id='T1',available_keyframe=8,large_extended=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'extents.json'
            def write(rows):
                path.write_text(json.dumps(dict(schema='farfield_observed_extent/v1',observations=rows)))
            write([row])
            self.assertEqual(set(ag.load_observations(path,['T1'])), {'T1'})
            with self.assertRaises(ValueError):
                ag.load_observations(path,['T1','T2'])
            write([row,row])
            with self.assertRaises(ValueError):
                ag.load_observations(path,['T1'])


if __name__ == '__main__':
    unittest.main()
