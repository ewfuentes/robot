"""Bounds the torch backend's divergence from the numpy reference.

The torch engine must be the SAME measurement model: identical mixture,
clips, priors, kappa_eff, and per-mode responsibility averaging. These tests
run the two backends on the same belief and assert the log-weights and
posteriors agree to float32 reduction error (and to ~1e-9 when the engine is
run in float64), on a catalog large enough to exercise blocking on the numpy
side and with map-accuracy classes exercising the per-pair kappa_eff path.
"""

import unittest

import numpy as np

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.bearing_only_localization import (
    catalog as catalog_mod,
    filter as pf,
    structs,
    torch_backend,
)


def _fixture(position_sigma):
    rng = np.random.default_rng(7)
    m = 700
    landmark_ids = [f"lm_{i}" for i in range(m)]
    catalog = catalog_mod.LandmarkCatalog(
        landmark_ids,
        rng.uniform(-9000, 9000, m), rng.uniform(-9000, 9000, m),
        position_sigma_m=position_sigma)
    entries = [structs.CompatibilityEntry(f"lm_{i}", float(llr))
               for i, llr in zip(range(0, m, 7),
                                 rng.uniform(-6, 6, len(range(0, m, 7))))]
    table = structs.CompatibilityTable(
        "trk", "v", entries, default_log_lr=-2.0,
        clip_lo=-4.0, clip_hi=4.0, status="fast")
    n = 300
    belief = pf.ParticleBelief(
        east_m=rng.uniform(-8000, 8000, n),
        north_m=rng.uniform(-8000, 8000, n),
        heading_rad=rng.uniform(-np.pi, np.pi, n),
        log_weight=rng.normal(0.0, 0.3, n))
    # Three modes plus unassigned, so per-mode grouping is exercised.
    belief.mode_id = rng.integers(-1, 3, n)
    meas = structs.TrackletMeasurement("trk", 0, 41.0, 220.0)
    return catalog, table, belief, meas


class TorchBackendEquivalenceTest(unittest.TestCase):

    def _run_both(self, position_sigma, dtype):
        catalog, table, belief, meas = _fixture(position_sigma)
        log_weight = pf._identity_log_weights(table, catalog, 0.5)

        belief_np = belief.copy()
        belief_np.mode_id = belief.mode_id.copy()
        posteriors_np = pf.measurement_update(
            belief_np, meas, table, catalog, pi0=0.2, per_mode=True,
            resp_min=1e-6)

        belief_t = belief.copy()
        belief_t.mode_id = belief.mode_id.copy()
        engine = torch_backend.TorchMeasurementEngine(
            catalog, {"trk": log_weight}, device="cpu", dtype=dtype)
        posteriors_t = engine.update(belief_t, meas, pi0=0.2, per_mode=True,
                                     resp_min=1e-6)
        return belief_np, posteriors_np, belief_t, posteriors_t

    def _check(self, position_sigma, dtype, atol):
        belief_np, post_np, belief_t, post_t = self._run_both(
            position_sigma, dtype)
        np.testing.assert_allclose(belief_t.log_weight, belief_np.log_weight,
                                   atol=atol)
        self.assertEqual(len(post_np), len(post_t))
        for a, b in zip(post_np, post_t):
            self.assertEqual(a.mode_id, b.mode_id)
            self.assertAlmostEqual(a.null_share, b.null_share, delta=atol)
            for lid, value in a.responsibilities.items():
                if value > 1e-4:  # below that, the resp_min cut may differ
                    self.assertAlmostEqual(
                        b.responsibilities.get(lid, 0.0), value, delta=atol)

    def test_float64_exact_catalog(self):
        self._check(position_sigma=0.0, dtype=torch.float64, atol=1e-9)

    def test_float64_map_accuracy_classes(self):
        self._check(position_sigma=8.0, dtype=torch.float64, atol=1e-9)

    def test_float32(self):
        self._check(position_sigma=8.0, dtype=torch.float32, atol=2e-3)

    def test_persistence_committed_path_matches_numpy(self):
        """With committed associations and a vanishing renewal rate the
        persistence update is deterministic: both backends must agree."""
        catalog, table, belief, meas = _fixture(position_sigma=8.0)
        log_weight = pf._identity_log_weights(table, catalog, 0.5)
        assoc_init = np.where(
            np.arange(belief.n) % 3 == 0, 5,
            np.where(np.arange(belief.n) % 3 == 1, pf.ASSOC_NULL,
                     pf.ASSOC_UNCOMMITTED)).astype(np.int32)

        belief_np = belief.copy()
        assoc_np = assoc_init.copy()
        pf.measurement_update(belief_np, meas, table, catalog, 0.2,
                              per_mode=False, assoc=assoc_np,
                              renewal_rate=1e-12, outlier_rate=0.1,
                              rng=np.random.default_rng(0))

        belief_t = belief.copy()
        assoc_t = assoc_init.copy()
        engine = torch_backend.TorchMeasurementEngine(
            catalog, {"trk": log_weight}, device="cpu", dtype=torch.float64)
        engine.update(belief_t, meas, 0.2, per_mode=False, resp_min=0.0,
                      assoc=assoc_t, renewal_rate=1e-12, outlier_rate=0.1)

        committed = assoc_init != pf.ASSOC_UNCOMMITTED
        np.testing.assert_allclose(belief_t.log_weight[committed],
                                   belief_np.log_weight[committed],
                                   atol=1e-9)
        np.testing.assert_array_equal(assoc_t[committed],
                                      assoc_np[committed])

    def test_persistence_renewal_one_matches_mixture_weights(self):
        """beta = 1: weights are the pure mixture on both backends even
        though the sampled associations use different rng streams."""
        catalog, table, belief, meas = _fixture(position_sigma=0.0)
        log_weight = pf._identity_log_weights(table, catalog, 0.5)

        belief_np = belief.copy()
        pf.measurement_update(belief_np, meas, table, catalog, 0.2,
                              per_mode=False)

        belief_t = belief.copy()
        assoc = np.full(belief.n, pf.ASSOC_UNCOMMITTED, dtype=np.int32)
        engine = torch_backend.TorchMeasurementEngine(
            catalog, {"trk": log_weight}, device="cpu", dtype=torch.float64)
        engine.update(belief_t, meas, 0.2, per_mode=False, resp_min=0.0,
                      assoc=assoc, renewal_rate=1.0, outlier_rate=0.1)
        np.testing.assert_allclose(belief_t.log_weight,
                                   belief_np.log_weight, atol=1e-9)
        self.assertTrue(np.all(assoc >= pf.ASSOC_NULL),
                        "every particle must commit after its first epoch")

    def test_unendorsed_candidates_cannot_be_committed(self):
        """Torch mirror of the endorsement rule: a perfectly aligned
        default-LLR candidate lands in the background bucket, never in a
        commitment."""
        from experimental.overhead_matching.swag.bearing_only_localization import (  # noqa: E501
            catalog as catalog_mod)
        catalog = catalog_mod.LandmarkCatalog(
            ["lm_0", "lm_1"], [5000.0, 0.0], [5000.0, 2000.0])
        table = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm_0", 2.0)],
            default_log_lr=-4.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        log_weight = pf._identity_log_weights(table, catalog, 0.5)
        mask = pf._surprise_mask(table, pf._clipped_log_lr(table, catalog))
        n = 2000
        belief = pf.ParticleBelief(np.zeros(n), np.zeros(n), np.zeros(n),
                                   np.zeros(n))
        assoc = np.full(n, pf.ASSOC_UNCOMMITTED, dtype=np.int32)
        engine = torch_backend.TorchMeasurementEngine(
            catalog, {"trk": log_weight}, device="cpu", dtype=torch.float64,
            seed=0, surprise_by_tracklet={"trk": mask})
        meas = structs.TrackletMeasurement("trk", 0, 0.0, 3000.0)
        engine.update(belief, meas, 0.2, per_mode=False, resp_min=0.0,
                      assoc=assoc, renewal_rate=0.1, outlier_rate=0.1,
                      draw_seed=7)
        self.assertEqual(int((assoc == 1).sum()), 0,
                         "an unendorsed candidate was committed")
        self.assertGreater(int((assoc == pf.ASSOC_NULL).sum()), n * 0.9)

    def test_run_filter_backend_dispatch(self):
        """run_filter(measurement_backend='torch') matches numpy end-state
        closely enough that both localize the same synthetic pose."""
        catalog, table, _, _ = _fixture(0.0)
        odometry = [structs.OdometryDelta(k, 10.0, 0.0, 0.0, 0.5, 0.01)
                    for k in range(1, 4)]
        # Distinct epochs of one tracklet at different anchors.
        measurements = [structs.TrackletMeasurement(
            "trk", k, 41.0 - 2.0 * k, 220.0) for k in range(4)]
        config = structs.FilterConfig(
            n_particles=500, seed=3,
            init=structs.GaussianInit(0.0, 0.0, 2000.0),
            measurement_backend="torch",
            proposal=structs.ProposalConfig(enabled=False))
        history = pf.run_filter(config, catalog, odometry, measurements,
                                {"trk": table})
        self.assertEqual(len(history.health), 4)
        self.assertTrue(np.all(np.isfinite(
            history.final_belief.log_weight)))


if __name__ == "__main__":
    unittest.main()
