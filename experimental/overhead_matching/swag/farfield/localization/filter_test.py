import math
import unittest
from unittest import mock

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    filter as pf,
    filter_catalog as catalog_mod,
    metrics,
    structs,
)


def _identity_table(tracklet_id: str, landmark_id: str,
                    clip: float = 4.0) -> structs.CompatibilityTable:
    return structs.CompatibilityTable(
        tracklet_id=tracklet_id,
        matcher_version="identity_stub_v1",
        entries=[structs.CompatibilityEntry(landmark_id, clip)],
        default_log_lr=-2.0,
        clip_lo=-clip,
        clip_hi=clip,
        status="fast")


def _catalog(ids, east, north, **kwargs):
    # max_visible_range_m is a required kwarg (filter_catalog); 10 km is the
    # old implicit default, so numerical behavior is unchanged.
    kwargs.setdefault("max_visible_range_m", 10000.0)
    return catalog_mod.LandmarkCatalog(ids, np.asarray(east, dtype=float),
                                       np.asarray(north, dtype=float),
                                       **kwargs)


class VonMisesTest(unittest.TestCase):
    """T-U1: density integrates to 1 across concentration regimes."""

    def test_normalizes(self):
        n_grid = 400000
        h = 2.0 * np.pi / n_grid
        theta = -np.pi + (np.arange(n_grid) + 0.5) * h
        for kappa in [1e-3, 1.0, 50.0, 3000.0]:
            total = float(np.sum(np.exp(pf.von_mises_logpdf(theta, kappa))) * h)
            self.assertAlmostEqual(total, 1.0, delta=1e-6,
                                   msg=f"kappa={kappa}")

    def test_concentration_limits(self):
        # kappa -> 0: uniform on the circle.
        self.assertAlmostEqual(float(pf.von_mises_logpdf(np.array(1.0), 1e-9)),
                               -math.log(2 * math.pi), places=6)
        # Large kappa: mode density matches Gaussian approx sqrt(kappa/2pi).
        mode = float(pf.von_mises_logpdf(np.array(0.0), 3000.0))
        self.assertAlmostEqual(mode, 0.5 * math.log(3000.0 / (2 * math.pi)),
                               delta=1e-3)

    def test_broadcasts_per_element_kappa(self):
        """kappa_eff varies per (particle, candidate), so the density must
        accept an array of concentrations."""
        delta = np.zeros((3, 2))
        kappa = np.array([[1.0, 100.0], [1.0, 100.0], [1.0, 100.0]])
        out = pf.von_mises_logpdf(delta, kappa)
        self.assertEqual(out.shape, (3, 2))
        for row in out:
            self.assertAlmostEqual(row[0],
                                   float(pf.von_mises_logpdf(np.array(0.0),
                                                             1.0)))


class MeasurementUpdateTest(unittest.TestCase):
    def test_bearing_convention(self):
        """Mini T-U3 at filter level: landmark due east, particle heading
        east sees body bearing 0; particle heading north sees +90."""
        belief = pf.ParticleBelief(
            east_m=np.zeros(2), north_m=np.zeros(2),
            heading_rad=np.array([math.pi / 2, 0.0]),  # east, north
            log_weight=np.zeros(2))
        meas = structs.TrackletMeasurement("trk_a", 0, 0.0, 3000.0)
        pf.measurement_update(belief, meas, _identity_table("trk_a", "lm_a"),
                              _catalog(["lm_a"], [1000.0], [0.0]), pi0=0.05)
        # Facing east -> predicted body bearing 0 -> matches measurement.
        # The mismatched particle is floored by the null hypothesis at
        # log(pi0 / 2pi), not driven to -inf (§5.3).
        self.assertGreater(belief.log_weight[0], belief.log_weight[1] + 5.0)
        self.assertAlmostEqual(float(belief.log_weight[1]),
                               math.log(0.05 / (2 * math.pi)), delta=0.01)

    def test_likelihood_is_a_normalized_density(self):
        """The mixture (pi0, (1-pi0) w_j) must integrate to 1 over the
        bearing circle for any pose. This is the invariant §5.3 warns is
        easy to break silently when the candidate prior changes."""
        catalog = _catalog(["lm_a", "lm_b", "lm_c"],
                           [1000.0, -500.0, 200.0],
                           [0.0, 800.0, -1500.0])
        table = structs.CompatibilityTable(
            "trk", "v",
            [structs.CompatibilityEntry("lm_a", 2.0),
             structs.CompatibilityEntry("lm_b", -1.0)],
            default_log_lr=0.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        n_grid = 20000
        step = 360.0 / n_grid
        total = 0.0
        for i in range(n_grid):
            belief = pf.ParticleBelief(np.array([10.0]), np.array([20.0]),
                                       np.array([0.3]), np.zeros(1))
            meas = structs.TrackletMeasurement(
                "trk", 0, -180.0 + (i + 0.5) * step, 40.0)
            pf.measurement_update(belief, meas, table, catalog, pi0=0.2)
            total += math.exp(float(belief.log_weight[0])) * math.radians(step)
        # The identity posterior (_identity_log_weights) sums to 1 over the
        # catalog, so the mixture integrates to exactly 1 over the circle.
        # (Under the earlier unnormalized w_j * LR_j weights the integral
        # was pi0 + (1-pi0) * sum w_j LR_j — a table-dependent constant
        # (~0.02 on the whole-map tables) that silently rescaled the
        # effective clutter share from 20% to ~92%.)
        self.assertAlmostEqual(total, 1.0, delta=1e-3)

    def test_null_floor_and_share(self):
        """Wildly inconsistent bearing: weights finite, null wins."""
        belief = pf.ParticleBelief(
            east_m=np.zeros(3), north_m=np.zeros(3),
            heading_rad=np.zeros(3), log_weight=np.zeros(3))
        # Landmark due north, but measurement says directly behind.
        meas = structs.TrackletMeasurement("trk_a", 0, 180.0, 3000.0)
        assoc = pf.measurement_update(
            belief, meas, _identity_table("trk_a", "lm_a"),
            _catalog(["lm_a"], [0.0], [1000.0]), pi0=0.05)[0]
        self.assertTrue(np.all(np.isfinite(belief.log_weight)))
        self.assertGreater(assoc.null_share, 0.99)

    def test_log_domain_extremes(self):
        """T-U5: 1000 candidates with LLRs at clip bounds, spanning several
        candidate blocks; no NaN/inf anywhere."""
        rng = np.random.default_rng(0)
        n, m = 50, 1000
        belief = pf.ParticleBelief(
            east_m=rng.uniform(-5000, 5000, n),
            north_m=rng.uniform(-5000, 5000, n),
            heading_rad=rng.uniform(-np.pi, np.pi, n),
            log_weight=np.zeros(n))
        landmark_ids = [f"lm_{i}" for i in range(m)]
        entries = [structs.CompatibilityEntry(
            lm_id, 4.0 if i % 2 == 0 else -4.0)
            for i, lm_id in enumerate(landmark_ids)]
        table = structs.CompatibilityTable(
            tracklet_id="trk", matcher_version="v", entries=entries,
            default_log_lr=-4.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        meas = structs.TrackletMeasurement("trk", 0, 37.0, 3000.0)
        catalog = _catalog(landmark_ids, rng.uniform(-20000, 20000, m),
                           rng.uniform(-20000, 20000, m))
        self.assertGreater(m, pf.CANDIDATE_BLOCK, "blocking not exercised")
        for pi0 in [0.05, 0.999]:
            b = belief.copy()
            assoc = pf.measurement_update(b, meas, table, catalog, pi0)[0]
            self.assertTrue(np.all(np.isfinite(b.log_weight)))
            self.assertTrue(math.isfinite(assoc.null_share))
            total = assoc.null_share + sum(assoc.responsibilities.values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_blocking_matches_unblocked(self):
        """Candidate blocking is a memory optimization only (A-8)."""
        rng = np.random.default_rng(3)
        m = 700
        landmark_ids = [f"lm_{i}" for i in range(m)]
        catalog = _catalog(landmark_ids, rng.uniform(-9000, 9000, m),
                           rng.uniform(-9000, 9000, m))
        table = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm_5", 3.0)],
            default_log_lr=-1.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        meas = structs.TrackletMeasurement("trk", 0, 12.0, 500.0)
        original_block = pf.CANDIDATE_BLOCK
        results = []
        try:
            for block in (64, 10_000):
                pf.CANDIDATE_BLOCK = block
                belief = pf.ParticleBelief(
                    np.linspace(-100, 100, 32), np.linspace(-50, 50, 32),
                    np.linspace(-1.0, 1.0, 32), np.zeros(32))
                assoc = pf.measurement_update(belief, meas, table, catalog,
                                              0.1)[0]
                results.append((belief.log_weight.copy(), assoc))
        finally:
            pf.CANDIDATE_BLOCK = original_block
        np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-12)
        self.assertAlmostEqual(results[0][1].null_share,
                               results[1][1].null_share, places=12)
        self.assertEqual(results[0][1].responsibilities.keys(),
                         results[1][1].responsibilities.keys())

    def test_llr_clipping(self):
        """Entries beyond the declared clip bounds are clipped (§6)."""
        belief_a = pf.ParticleBelief(
            east_m=np.zeros(1), north_m=np.zeros(1),
            heading_rad=np.zeros(1), log_weight=np.zeros(1))
        belief_b = belief_a.copy()
        meas = structs.TrackletMeasurement("trk", 0, 0.0, 100.0)
        catalog = _catalog(["lm"], [0.0], [1000.0])
        table_huge = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm", 1000.0)],
            default_log_lr=-2.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        table_clip = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm", 4.0)],
            default_log_lr=-2.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        pf.measurement_update(belief_a, meas, table_huge, catalog, 0.05)
        pf.measurement_update(belief_b, meas, table_clip, catalog, 0.05)
        self.assertAlmostEqual(float(belief_a.log_weight[0]),
                               float(belief_b.log_weight[0]), places=12)

    def test_kappa_is_capped(self):
        """A matcher (or a tracker) handing over an implausible kappa must
        not get unbounded authority over the posterior."""
        catalog = _catalog(["lm"], [0.0], [1000.0])
        table = _identity_table("trk", "lm")
        weights = []
        for kappa in (pf.MAX_KAPPA, pf.MAX_KAPPA * 1e6):
            belief = pf.ParticleBelief(np.zeros(1), np.zeros(1), np.zeros(1),
                                       np.zeros(1))
            pf.measurement_update(
                belief, structs.TrackletMeasurement("trk", 0, 0.0, kappa),
                table, catalog, 0.05)
            weights.append(float(belief.log_weight[0]))
        self.assertAlmostEqual(weights[0], weights[1], places=9)

    def test_rejects_invalid_kappa(self):
        catalog = _catalog(["lm"], [0.0], [1000.0])
        belief = pf.ParticleBelief(np.zeros(1), np.zeros(1), np.zeros(1),
                                   np.zeros(1))
        for kappa in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                pf.measurement_update(
                    belief, structs.TrackletMeasurement("t", 0, 0.0, kappa),
                    _identity_table("t", "lm"), catalog, 0.05)


class KappaEffTest(unittest.TestCase):
    def test_map_error_widens_the_bearing_likelihood(self):
        """kappa_eff combines tracklet kappa with projected map error (§4):
        a 50 m position error at 1 km is ~2.9 deg of bearing."""
        catalog = _catalog(["lm"], [0.0], [1000.0], position_sigma_m=50.0)
        kappa_z = 1.0 / math.radians(1.0) ** 2
        kappa_eff = catalog.kappa_eff(kappa_z, np.array([[1000.0]]))
        sigma_eff_deg = math.degrees(1.0 / math.sqrt(float(kappa_eff[0, 0])))
        self.assertAlmostEqual(sigma_eff_deg, math.hypot(1.0, math.degrees(
            50.0 / 1000.0)), places=3)

    def test_exact_catalog_leaves_kappa_untouched(self):
        catalog = _catalog(["lm"], [0.0], [1000.0])
        kappa_eff = catalog.kappa_eff(500.0, np.array([[1000.0], [50.0]]))
        np.testing.assert_allclose(kappa_eff, 500.0)

    def test_rejects_mismatched_arrays(self):
        with self.assertRaises(ValueError):
            catalog_mod.LandmarkCatalog(["a", "b"], np.array([1.0]),
                                        np.array([2.0]),
                                        max_visible_range_m=10000.0)
        with self.assertRaises(ValueError):
            catalog_mod.LandmarkCatalog(["a", "a"], np.array([1.0, 2.0]),
                                        np.array([1.0, 2.0]),
                                        max_visible_range_m=10000.0)


class MotionModelTest(unittest.TestCase):
    """Hand-derived fixtures for forward/left/CW-yaw propagation (§5.2).

    Hand-computed, not generated by the code under test (the T-U3
    rationale): motion-side sign bugs are the same class as bearing-side
    ones. Heading is compass (CW from north): forward -> (sin h, cos h),
    left -> (-cos h, sin h); yaw updates before translation.
    """

    @staticmethod
    def _step(heading_deg, forward, left, dyaw_deg,
              sigma_m=0.0, sigma_yaw_deg=0.0, n=1, seed=0):
        belief = pf.ParticleBelief(
            east_m=np.zeros(n), north_m=np.zeros(n),
            heading_rad=np.full(n, math.radians(heading_deg)),
            log_weight=np.zeros(n))
        delta = structs.OdometryDelta(
            keyframe_idx=1, forward_m=forward, left_m=left,
            delta_yaw_cw_rad=math.radians(dyaw_deg), sigma_m=sigma_m,
            sigma_yaw_rad=math.radians(sigma_yaw_deg))
        pf.motion_update(belief, delta, 0.0, np.random.default_rng(seed))
        return belief

    def _assert_pose(self, belief, east, north, heading_deg):
        self.assertAlmostEqual(float(belief.east_m[0]), east, places=9)
        self.assertAlmostEqual(float(belief.north_m[0]), north, places=9)
        self.assertAlmostEqual(
            math.degrees(float(belief.heading_rad[0])) % 360.0,
            heading_deg % 360.0, places=9)

    def test_straight_north(self):
        self._assert_pose(self._step(0.0, 10.0, 0.0, 0.0), 0.0, 10.0, 0.0)

    def test_straight_east(self):
        self._assert_pose(self._step(90.0, 10.0, 0.0, 0.0), 10.0, 0.0, 90.0)

    def test_left_of_east_is_north(self):
        self._assert_pose(self._step(90.0, 0.0, 5.0, 0.0), 0.0, 5.0, 90.0)

    def test_pure_rotation(self):
        self._assert_pose(self._step(0.0, 0.0, 0.0, 30.0), 0.0, 0.0, 30.0)

    def test_corner_rotates_then_moves_in_updated_frame(self):
        belief = self._step(0.0, 10.0, 0.0, 90.0)
        self._assert_pose(belief, 10.0, 0.0, 90.0)

    def test_heading_noise_creates_cross_track_spread(self):
        """Yaw noise sampled before rotation creates cross-track spread."""
        belief = self._step(0.0, 100.0, 0.0, 0.0, sigma_yaw_deg=5.0, n=40000)
        self.assertAlmostEqual(float(np.std(belief.east_m)),
                               100.0 * math.radians(5.0),
                               delta=0.15)
        self.assertAlmostEqual(
            float(np.std(np.degrees(belief.heading_rad))), 5.0, delta=0.1)


class ResamplerTest(unittest.TestCase):
    """T-U6: systematic resampler preserves expected weights."""

    def test_unbiased(self):
        weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        east = np.arange(5, dtype=np.float64)
        counts = np.zeros(5)
        n_trials = 2000
        for trial in range(n_trials):
            belief = pf.ParticleBelief(
                east_m=east.copy(), north_m=np.zeros(5),
                heading_rad=np.zeros(5), log_weight=np.log(weights))
            pf.systematic_resample(belief, np.random.default_rng(trial),
                                   regularization=0.0)
            for i in range(5):
                counts[i] += np.sum(belief.east_m == east[i])
        freq = counts / (n_trials * 5)
        np.testing.assert_allclose(freq, weights, atol=0.02)

    def test_regularization_preserves_mean_and_inflates_spread(self):
        rng = np.random.default_rng(0)
        n = 20000
        belief = pf.ParticleBelief(
            east_m=rng.normal(100.0, 30.0, n), north_m=rng.normal(-50.0, 30.0, n),
            heading_rad=rng.normal(0.2, 0.05, n), log_weight=np.zeros(n))
        before_mean = metrics.mean_pose(belief)
        before_std = metrics.position_std_m(belief)
        pf.systematic_resample(belief, rng, regularization=1.0)
        after_mean = metrics.mean_pose(belief)
        self.assertAlmostEqual(before_mean[0], after_mean[0], delta=2.0)
        self.assertAlmostEqual(before_mean[1], after_mean[1], delta=2.0)
        # Bandwidth is sigma * n^(-1/6): variance grows by 1 + n^(-1/3).
        expected = before_std * math.sqrt(1.0 + n ** (-1.0 / 3.0))
        self.assertAlmostEqual(metrics.position_std_m(belief), expected, delta=1.5)

    def test_uniform_weights_ess(self):
        log_w = np.zeros(100)
        self.assertAlmostEqual(pf.ess(log_w), 100.0, places=6)
        log_w = np.full(100, -np.inf)
        log_w[0] = 0.0
        self.assertAlmostEqual(pf.ess(log_w), 1.0, places=6)

    @staticmethod
    def _belief_with_tiny_cluster(group_mass=4e-6, n=1000):
        """A 4-particle proposal cluster holding `group_mass`, the rest
        diffuse — mass * n far below one offspring, so plain stratified
        allocation rounds the cluster to extinction."""
        rng = np.random.default_rng(0)
        east = np.concatenate([np.zeros(4), rng.normal(5000.0, 100.0, n - 4)])
        weights = np.concatenate([
            np.full(4, group_mass / 4),
            np.full(n - 4, (1.0 - group_mass) / (n - 4))])
        belief = pf.ParticleBelief(
            east_m=east, north_m=np.zeros(n), heading_rad=np.zeros(n),
            log_weight=np.log(weights))
        belief.proposal_event_id[:4] = 0
        belief.proposal_hypothesis[:4] = 7
        return belief

    def test_survival_floor_keeps_hypothesis_group_alive(self):
        belief = self._belief_with_tiny_cluster()
        pf.systematic_resample(belief, np.random.default_rng(1),
                               regularization=0.0, survival_floor=16)
        survivors = belief.proposal_event_id == 0
        self.assertGreaterEqual(int(survivors.sum()), 16)
        # Representation is guaranteed; probability is not: the group's
        # offspring carry exactly its pre-resample posterior mass.
        held = float(belief.normalized_weights()[survivors].sum())
        self.assertAlmostEqual(held, 4e-6, delta=1e-8)

    def test_survival_floor_zero_reproduces_legacy_extinction(self):
        belief = self._belief_with_tiny_cluster()
        pf.systematic_resample(belief, np.random.default_rng(1),
                               regularization=0.0, survival_floor=0)
        self.assertEqual(int((belief.proposal_event_id == 0).sum()), 0)
        np.testing.assert_array_equal(belief.log_weight, np.zeros(belief.n))

    def test_survival_floor_respects_min_mass(self):
        """A genuinely refuted group (mass below the min) may still die."""
        belief = self._belief_with_tiny_cluster(group_mass=4e-6)
        pf.systematic_resample(belief, np.random.default_rng(1),
                               regularization=0.0, survival_floor=16,
                               survival_min_mass=1e-3)
        self.assertEqual(int((belief.proposal_event_id == 0).sum()), 0)


class ProposalInjectionInvariantTest(unittest.TestCase):
    def test_actual_injection_count_owns_retention_and_mass(self):
        belief = pf.ParticleBelief(
            east_m=np.arange(10, dtype=float),
            north_m=np.zeros(10), heading_rad=np.zeros(10),
            log_weight=np.zeros(10))
        config = structs.FilterConfig(
            n_particles=10, seed=0,
            init=structs.GaussianInit(0.0, 0.0, 1.0),
            proposal=structs.ProposalConfig(inject_fraction=0.5))
        result = type("Result", (), {
            "hypotheses": [object()], "event_id": 7})()
        returned = (
            np.array([100.0, 200.0]), np.array([300.0, 400.0]),
            np.array([0.1, 0.2]), np.array([0, 0], dtype=np.int64))

        with mock.patch.object(
                pf.proposal_mod, "sample_particles", return_value=returned):
            n_injected, kept = pf.inject_proposal(
                belief, result, config, np.random.default_rng(2))

        self.assertEqual(n_injected, 2)
        self.assertEqual(kept.size, 8)
        self.assertEqual(belief.n, 10)
        self.assertAlmostEqual(
            float(np.exp(belief.log_weight[:8]).sum()), 0.8)
        self.assertAlmostEqual(
            float(np.exp(belief.log_weight[8:]).sum()), 0.2)
        np.testing.assert_array_equal(belief.proposal_event_id[8:], [7, 7])



class PerModeAssociationTest(unittest.TestCase):
    """§5.4 `[CONTRACT]`: two modes with contradictory explanations must be
    reported separately. The whole-belief average describes neither."""

    def test_modes_report_contradictory_associations(self):
        # Two clusters, mirrored east/west, each seeing "its" landmark dead
        # ahead: mode 0 must attribute the bearing to lm_west, mode 1 to
        # lm_east, and the global average must sit between the two.
        east = np.concatenate([np.full(500, -1000.0), np.full(500, 1000.0)])
        belief = pf.ParticleBelief(
            east_m=east, north_m=np.zeros(1000),
            heading_rad=np.zeros(1000), log_weight=np.zeros(1000),
            mode_id=np.concatenate([np.zeros(500, dtype=np.int64),
                                    np.ones(500, dtype=np.int64)]))
        catalog = _catalog(["lm_west", "lm_east"], [-1000.0, 1000.0],
                           [3000.0, 3000.0])
        table = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm_west", 2.0),
                         structs.CompatibilityEntry("lm_east", 2.0)],
            default_log_lr=-2.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        posteriors = pf.measurement_update(
            belief, structs.TrackletMeasurement("trk", 0, 0.0, 2000.0),
            table, catalog, pi0=0.05)

        by_mode = {p.mode_id: p for p in posteriors}
        self.assertEqual(set(by_mode), {None, 0, 1})
        self.assertGreater(by_mode[0].responsibilities["lm_west"], 0.9)
        self.assertLess(by_mode[0].responsibilities["lm_east"], 0.05)
        self.assertGreater(by_mode[1].responsibilities["lm_east"], 0.9)
        self.assertLess(by_mode[1].responsibilities["lm_west"], 0.05)
        # The global average is the number that describes neither mode.
        self.assertAlmostEqual(by_mode[None].responsibilities["lm_west"], 0.5,
                               delta=0.1)
        for posterior in posteriors:
            total = posterior.null_share + sum(
                posterior.responsibilities.values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_per_mode_can_be_disabled(self):
        belief = pf.ParticleBelief(np.zeros(4), np.zeros(4), np.zeros(4),
                                   np.zeros(4),
                                   mode_id=np.zeros(4, dtype=np.int64))
        posteriors = pf.measurement_update(
            belief, structs.TrackletMeasurement("trk", 0, 0.0, 100.0),
            _identity_table("trk", "lm"), _catalog(["lm"], [0.0], [900.0]),
            pi0=0.05, per_mode=False)
        self.assertEqual(len(posteriors), 1)
        self.assertIsNone(posteriors[0].mode_id)


# MapPoseTest moved to metrics_test.py with the metrics split.


def _tiny_inputs():
    catalog = _catalog(["lm_a", "lm_b"], [2000.0, -1500.0], [1000.0, 2500.0])
    odometry = [structs.OdometryDelta(keyframe_idx=k, forward_m=10.0,
                                      left_m=0.0, delta_yaw_cw_rad=0.0, sigma_m=0.5,
                                      sigma_yaw_rad=math.radians(2.0))
                for k in range(1, 8)]
    measurements = [
        structs.TrackletMeasurement("trk_lm_a", 2, 15.0, 3000.0),
        structs.TrackletMeasurement("trk_lm_b", 5, -40.0, 3000.0)]
    tables = {"trk_lm_a": _identity_table("trk_lm_a", "lm_a"),
              "trk_lm_b": _identity_table("trk_lm_b", "lm_b")}
    return catalog, odometry, measurements, tables


def _tiny_config(seed: int = 7, **kwargs) -> structs.FilterConfig:
    return structs.FilterConfig(
        n_particles=500, seed=seed,
        init=structs.GaussianInit(0.0, 0.0, 300.0), **kwargs)


class ValidationTest(unittest.TestCase):
    def _inputs(self):
        return _tiny_inputs()

    def _config(self, seed: int = 7, **kwargs) -> structs.FilterConfig:
        return _tiny_config(seed, **kwargs)

    def test_rejects_duplicate_information_epoch(self):
        """T-F1: re-submitting a tracklet's epoch double-counts evidence."""
        catalog, odometry, measurements, tables = self._inputs()
        with self.assertRaisesRegex(ValueError, "duplicate measurement"):
            pf.run_filter(self._config(), catalog, odometry,
                          measurements + measurements[:1], tables)

    def test_rejects_missing_table(self):
        catalog, odometry, measurements, tables = self._inputs()
        with self.assertRaisesRegex(ValueError, "no CompatibilityTable"):
            pf.run_filter(self._config(), catalog, odometry, measurements,
                          {"trk_lm_a": tables["trk_lm_a"]})

    def test_rejects_out_of_range_anchor(self):
        catalog, odometry, measurements, tables = self._inputs()
        stray = structs.TrackletMeasurement("trk_lm_a", 99, 0.0, 100.0)
        with self.assertRaisesRegex(ValueError, "outside"):
            pf.run_filter(self._config(), catalog, odometry, [stray], tables)

    def test_rejects_bad_config(self):
        catalog, odometry, measurements, tables = self._inputs()
        for kwargs in [dict(pi0=0.0), dict(pi0=1.0), dict(checkpoint_every=0)]:
            with self.assertRaises(ValueError):
                pf.run_filter(self._config(**kwargs), catalog, odometry,
                              measurements, tables)
        with self.assertRaises(ValueError):
            pf.run_filter(
                structs.FilterConfig(
                    n_particles=0, seed=1,
                    init=structs.GaussianInit(0.0, 0.0, 1.0)),
                catalog, odometry, measurements, tables)

    def test_rejects_out_of_order_odometry(self):
        catalog, odometry, measurements, tables = self._inputs()
        with self.assertRaisesRegex(ValueError, "out of order"):
            pf.run_filter(self._config(), catalog, list(reversed(odometry)),
                          measurements, tables)

    def test_rejects_negative_sigma_yaw(self):
        catalog, odometry, measurements, tables = self._inputs()
        bad = [structs.OdometryDelta(keyframe_idx=1, forward_m=1.0,
                                     left_m=0.0, delta_yaw_cw_rad=0.0, sigma_m=1.0,
                                     sigma_yaw_rad=-0.1)]
        with self.assertRaisesRegex(ValueError, "sigma_yaw_rad"):
            pf.run_filter(self._config(), catalog, bad, [], tables)

    def test_rejects_non_finite_increment(self):
        catalog, odometry, measurements, tables = self._inputs()
        bad = [structs.OdometryDelta(keyframe_idx=1,
                                     forward_m=float("nan"), left_m=0.0,
                                     delta_yaw_cw_rad=0.0, sigma_m=1.0,
                                     sigma_yaw_rad=0.1)]
        with self.assertRaisesRegex(ValueError, "not finite"):
            pf.run_filter(self._config(), catalog, bad, [], tables)

    def test_rejects_inverted_clip_bounds(self):
        catalog, odometry, measurements, tables = self._inputs()
        bad = structs.CompatibilityTable(
            "trk_lm_a", "v", [], default_log_lr=0.0, clip_lo=4.0,
            clip_hi=-4.0, status="fast")
        tables = dict(tables, trk_lm_a=bad)
        with self.assertRaisesRegex(ValueError, "clip_lo"):
            pf.run_filter(self._config(), catalog, odometry, measurements,
                          tables)


class RunFilterDeterminismTest(unittest.TestCase):
    """T-U7: same (config, seed, input log) -> bit-identical history hash.

    In-process only: this pins reproducibility for a given environment, not
    across numpy/BLAS versions (see run_io.py's replay note).
    """

    def test_same_seed_same_hash(self):
        args = _tiny_inputs()
        h1 = pf.run_filter(_tiny_config(7), *args)
        h2 = pf.run_filter(_tiny_config(7), *args)
        self.assertEqual(h1.particle_history_sha256,
                         h2.particle_history_sha256)

    def test_different_seed_different_hash(self):
        args = _tiny_inputs()
        h1 = pf.run_filter(_tiny_config(7), *args)
        h2 = pf.run_filter(_tiny_config(8), *args)
        self.assertNotEqual(h1.particle_history_sha256,
                            h2.particle_history_sha256)


class GroupBandwidthTest(unittest.TestCase):
    """Kernel bandwidth is per group (mode / proposal provenance / diffuse):
    one global bandwidth re-diffuses every converged cluster whenever the
    belief is spread — the whole-map harbor failure mode."""

    def _two_component_belief(self, rng):
        n_tight, n_diffuse = 2000, 18000
        belief = pf.ParticleBelief(
            east_m=np.concatenate([rng.normal(0.0, 10.0, n_tight),
                                   rng.uniform(-12000, 12000, n_diffuse)]),
            north_m=np.concatenate([rng.normal(0.0, 10.0, n_tight),
                                    rng.uniform(-12000, 12000, n_diffuse)]),
            heading_rad=np.concatenate([
                rng.normal(1.0, 0.02, n_tight),
                rng.uniform(-np.pi, np.pi, n_diffuse)]),
            log_weight=np.concatenate([
                np.full(n_tight, math.log(0.5 / n_tight)),
                np.full(n_diffuse, math.log(0.5 / n_diffuse))]))
        return belief, n_tight

    def _check_cluster_survives(self, belief):
        member = np.hypot(belief.east_m, belief.north_m) < 500.0
        # The cluster held half the posterior mass, so it should keep about
        # half the particles...
        self.assertGreater(float(member.mean()), 0.35)
        # ...and stay tight: its own bandwidth is ~3 m. A global bandwidth
        # would be ~900 m and smear it past recognition.
        self.assertLess(float(belief.east_m[member].std()), 40.0)
        self.assertLess(float(
            np.std(belief.heading_rad[member] - 1.0)), 0.1)

    def test_tight_mode_survives_resampling_in_spread_belief(self):
        rng = np.random.default_rng(1)
        belief, n_tight = self._two_component_belief(rng)
        belief.mode_id = np.concatenate([
            np.zeros(n_tight, dtype=np.int64),
            np.full(belief.n - n_tight, -1, dtype=np.int64)])
        pf.systematic_resample(belief, rng, regularization=1.0)
        self._check_cluster_survives(belief)

    def test_injected_cluster_survives_before_becoming_a_mode(self):
        """Freshly injected hypothesis clusters have no mode yet; their
        proposal provenance is their bandwidth group."""
        rng = np.random.default_rng(2)
        belief, n_tight = self._two_component_belief(rng)
        belief.proposal_event_id = np.concatenate([
            np.zeros(n_tight, dtype=np.int64),
            np.full(belief.n - n_tight, -1, dtype=np.int64)])
        belief.proposal_hypothesis = np.concatenate([
            np.full(n_tight, 3, dtype=np.int64),
            np.full(belief.n - n_tight, -1, dtype=np.int64)])
        pf.systematic_resample(belief, rng, regularization=1.0)
        self._check_cluster_survives(belief)

    def test_unimodal_belief_matches_global_rule(self):
        """With a single group the per-group rule IS the old global rule."""
        rng = np.random.default_rng(0)
        n = 20000
        belief = pf.ParticleBelief(
            east_m=rng.normal(100.0, 30.0, n),
            north_m=rng.normal(-50.0, 30.0, n),
            heading_rad=rng.normal(0.2, 0.05, n), log_weight=np.zeros(n))
        before_std = metrics.position_std_m(belief)
        pf.systematic_resample(belief, rng, regularization=1.0)
        expected = before_std * math.sqrt(1.0 + n ** (-1.0 / 3.0))
        self.assertAlmostEqual(metrics.position_std_m(belief), expected, delta=1.5)


class AssociationPersistenceTest(unittest.TestCase):
    """§5.3 persistence: a tracklet pays its identity prior once, then its
    epochs are pure geometry (the anti-dilution property, audit A-5)."""

    def setUp(self):
        rng = np.random.default_rng(0)
        m = 400
        east = rng.uniform(-9000, 9000, m)
        north = rng.uniform(-9000, 9000, m)
        east[0], north[0] = 0.0, 2000.0  # the true landmark
        self.catalog = _catalog([f"lm_{i}" for i in range(m)], east, north)
        self.table = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm_0", 2.0)],
            default_log_lr=-4.0, clip_lo=-4.0, clip_hi=4.0, status="fast")

    @staticmethod
    def _belief():
        # Particle 0 at the origin heading north (sees lm_0 dead ahead);
        # particle 1 in empty water.
        return pf.ParticleBelief(np.array([0.0, 7000.0]),
                                 np.array([0.0, -8500.0]),
                                 np.array([0.0, 0.0]), np.zeros(2))

    def _meas(self, epoch):
        return structs.TrackletMeasurement("trk", epoch, 0.0, 3000.0)

    def test_renewal_rate_one_reduces_to_the_mixture(self):
        """beta = 1 is per-epoch marginalization exactly: the persistence
        model nests the old one."""
        belief_mixture = self._belief()
        pf.measurement_update(belief_mixture, self._meas(0), self.table,
                              self.catalog, 0.2)
        belief_persist = self._belief()
        assoc = np.full(2, pf.ASSOC_UNCOMMITTED, dtype=np.int32)
        pf.measurement_update(belief_persist, self._meas(0), self.table,
                              self.catalog, 0.2, assoc=assoc,
                              renewal_rate=1.0,
                              rng=np.random.default_rng(1))
        np.testing.assert_allclose(belief_persist.log_weight,
                                   belief_mixture.log_weight, atol=1e-12)

    def test_committed_epochs_compound(self):
        """After the first epoch commits, each further epoch is worth
        ~log(vM_peak / (1/2pi)) — independent of catalog size — instead of
        the diluted mixture nudge."""
        rng = np.random.default_rng(3)
        belief = self._belief()
        assoc = np.full(2, pf.ASSOC_UNCOMMITTED, dtype=np.int32)
        for epoch in range(6):
            pf.measurement_update(belief, self._meas(epoch), self.table,
                                  self.catalog, 0.2, assoc=assoc,
                                  renewal_rate=0.05, rng=rng)
        self.assertEqual(assoc[0], 0, "truth particle did not commit to "
                                      "the true landmark")
        self.assertEqual(assoc[1], pf.ASSOC_NULL,
                         "empty-water particle should call the tracklet "
                         "clutter")
        advantage = belief.log_weight[0] - belief.log_weight[1]
        self.assertGreater(advantage, 10.0,
                           "committed epochs are not compounding")

    def test_wrong_commitment_renews_away(self):
        """A bad early commit is not a life sentence: the renewal branch
        wins whenever the committed geometry stops explaining the bearing."""
        n = 200
        belief = pf.ParticleBelief(np.zeros(n), np.zeros(n), np.zeros(n),
                                   np.zeros(n))
        assoc = np.full(n, 5, dtype=np.int32)  # committed to a wrong lm
        rng = np.random.default_rng(4)
        for epoch in range(4):
            pf.measurement_update(belief, self._meas(epoch), self.table,
                                  self.catalog, 0.2, assoc=assoc,
                                  renewal_rate=0.3, rng=rng)
        self.assertGreater(float(np.mean(assoc == 0)), 0.8)

    def test_unendorsed_candidates_cannot_be_committed(self):
        """A perfectly aligned default-LLR candidate must land in the
        background bucket (ASSOC_NULL), never be committed: committing to
        the best-aligned of hundreds of in-cone unendorsed candidates and
        riding full vM concentration is data-association overfitting (the
        whole-map drift failure). The mixture still counts it exactly."""
        # lm_1 sits dead ahead of the particle but is NOT a table entry.
        catalog = _catalog(["lm_0", "lm_1"], [5000.0, 0.0], [5000.0, 2000.0])
        table = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm_0", 2.0)],
            default_log_lr=-4.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        n = 300
        belief = pf.ParticleBelief(np.zeros(n), np.zeros(n), np.zeros(n),
                                   np.zeros(n))
        assoc = np.full(n, pf.ASSOC_UNCOMMITTED, dtype=np.int32)
        pf.measurement_update(belief, self._meas(0), table, catalog, 0.2,
                              assoc=assoc, renewal_rate=0.1,
                              rng=np.random.default_rng(6))
        self.assertEqual(int(np.sum(assoc == 1)), 0,
                         "an unendorsed candidate was committed")
        self.assertGreater(float(np.mean(assoc == pf.ASSOC_NULL)), 0.9)

    def test_take_reindexes_associations(self):
        belief = self._belief()
        belief.associations["trk"] = np.array([7, 9], dtype=np.int32)
        belief.take(np.array([1, 1]))
        np.testing.assert_array_equal(belief.associations["trk"], [9, 9])


class _FakeTracker:
    def __init__(self, records):
        self._previous = {r.mode_id: r for r in records}


def _mode_record(mode_id, east, north, heading_deg):
    return structs.ModeRecord(
        mode_id=mode_id, weight=0.9, n_particles=1000,
        mean_east_m=east, mean_north_m=north, mean_heading_deg=heading_deg,
        position_std_m=30.0, heading_std_deg=3.0, birth_keyframe_idx=0)


class EvidenceGateTest(unittest.TestCase):
    """§5.5 evidence gate: injection must beat the best existing mode's
    explanation of the window, not merely be triggered by distress."""

    def setUp(self):
        from experimental.overhead_matching.swag.farfield.localization import (  # noqa: E501
            proposal as proposal_mod)
        self.proposal_mod = proposal_mod
        self.catalog = _catalog(["lm_true", "lm_far"],
                                [0.0, 5000.0], [1000.0, -3000.0])
        self.table = structs.CompatibilityTable(
            tracklet_id="trk", matcher_version="v",
            entries=[structs.CompatibilityEntry("lm_true", 4.0),
                     structs.CompatibilityEntry("lm_far", 4.0)],
            default_log_lr=-2.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        # The vehicle at the origin, heading north, sees lm_true dead ahead.
        self.window = [structs.TrackletMeasurement("trk", 5, 0.0, 820.0)]
        self.config = structs.FilterConfig(
            n_particles=10, seed=0,
            init=structs.GaussianInit(0.0, 0.0, 1.0))

        def score_fn(east, north, heading, meas):
            return pf.pose_log_likelihood(east, north, heading, meas,
                                          self.table, self.catalog, pi0=0.2)
        self.score_fn = score_fn

    def _disc_hypothesis(self, landmark_id):
        index = self.catalog.index_of(landmark_id)
        return self.proposal_mod.VisibilityDiscHypothesis(
            kind=self.proposal_mod.SINGLE, tracklet_ids=("trk",),
            landmark_ids=(landmark_id,),
            landmark=(float(self.catalog.east_m[index]),
                      float(self.catalog.north_m[index])),
            bearing_rad=0.0,
            max_range_m=2000.0)

    def _result(self, hypotheses):
        return self.proposal_mod.ProposalResult(
            event_id=0, keyframe_idx=5, trigger="null_share",
            hypotheses=hypotheses, particle_budget=2,
            n_tracklets_considered=1, n_combinations_total=1,
            n_combinations_enumerated=1, n_combinations_sampled=0,
            n_combinations_geometry_pruned=0,
            n_partially_represented_ties=0,
            n_solution_clusters_merged=0,
            represented_compatibility_mass=1.0)

    @staticmethod
    def _belief_at(east, north, heading_rad):
        return pf.ParticleBelief(np.full(4, east), np.full(4, north),
                                 np.full(4, heading_rad), np.zeros(4))

    def test_no_modes_always_passes(self):
        passed, _, _ = pf._evidence_gate(
            _FakeTracker([]), self._result([self._disc_hypothesis("lm_far")]),
            self.window, self.config, np.random.default_rng(0),
            self.score_fn, self._belief_at(0.0, 0.0, 0.0), self.catalog)
        self.assertTrue(passed)

    def test_equivalent_alternative_is_rejected(self):
        """The belief already explains the bearing (via lm_true); a disc
        around the tied lm_far explains it no better, so displacing half
        the belief for it is unjustified."""
        tracker = _FakeTracker([_mode_record(0, 0.0, 0.0, 0.0)])
        passed, best, ref = pf._evidence_gate(
            tracker, self._result([self._disc_hypothesis("lm_far")]),
            self.window, self.config, np.random.default_rng(0),
            self.score_fn, self._belief_at(0.0, 0.0, 0.0), self.catalog)
        self.assertFalse(passed)
        self.assertLess(best, ref + 1.0)

    def test_kidnapped_mode_is_beaten(self):
        """A displaced belief explains nothing; the hypothesis explains the
        bearing exactly — recovery must proceed."""
        tracker = _FakeTracker([_mode_record(0, 9000.0, 9000.0, 90.0)])
        passed, best, ref = pf._evidence_gate(
            tracker, self._result([self._disc_hypothesis("lm_true")]),
            self.window, self.config, np.random.default_rng(0),
            self.score_fn, self._belief_at(9000.0, 9000.0, math.pi / 2),
            self.catalog)
        self.assertTrue(passed)
        self.assertGreater(best, ref + 1.0)

    def test_reference_reads_the_committed_state(self):
        """The reference scores the incumbent AS THE FILTER SCORES IT: a
        belief committed to the true landmark must out-score the same
        belief committed to a wrong one — the plain mixture at the same
        poses cannot tell them apart, which is how the gate passed junk
        injections against a converged mode (whole-map kf 280-330)."""
        rng = np.random.default_rng(8)
        m = 400
        east = rng.uniform(-9000, 9000, m)
        north = rng.uniform(-9000, 9000, m)
        east[0], north[0] = 0.0, 1000.0  # lm_true's geometry
        dense = _catalog([f"lm_{i}" for i in range(m)], east, north)
        table = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm_0", 4.0)],
            default_log_lr=-4.0, clip_lo=-4.0, clip_hi=4.0, status="fast")

        def score_fn(e, n, h, meas):
            return pf.pose_log_likelihood(e, n, h, meas, table, dense,
                                          pi0=0.2)

        belief_true = self._belief_at(0.0, 0.0, 0.0)
        belief_true.associations["trk"] = np.zeros(4, dtype=np.int32)
        belief_wrong = self._belief_at(0.0, 0.0, 0.0)
        belief_wrong.associations["trk"] = np.full(4, 7, dtype=np.int32)
        true_ref = pf._belief_window_reference(
            belief_true, self.window, self.config, dense, score_fn)
        wrong_ref = pf._belief_window_reference(
            belief_wrong, self.window, self.config, dense, score_fn)
        self.assertGreater(true_ref, wrong_ref + 2.0)


class RangeCapTest(unittest.TestCase):
    """The extractor's distance bucket as a one-sided cap on each landmark
    component: identity inside the cap, a Gaussian tail beyond it, never a
    lower bound, and exactly the old likelihood when disabled."""

    def test_log_term_shape(self):
        r = np.array([10.0, 100.0, 125.0, 200.0])
        term = pf.range_cap_log_term(r, 100.0, 0.25)
        np.testing.assert_allclose(term, [0.0, 0.0, -0.5, -8.0])
        self.assertEqual(pf.range_cap_log_term(r, None, 0.25), 0.0)
        with self.assertRaises(ValueError):
            pf.range_cap_log_term(r, 0.0, 0.25)
        with self.assertRaises(ValueError):
            pf.range_cap_log_term(r, 100.0, 0.0)

    @staticmethod
    def _meas(range_max_m, anchor=0):
        return structs.TrackletMeasurement(
            tracklet_id="T1", anchor_keyframe_idx=anchor,
            bearing_forward_cw_deg=0.0,
            kappa=1.0 / math.radians(2.0) ** 2, range_max_m=range_max_m)

    def test_cap_penalises_only_particles_beyond_it(self):
        # Two poses heading north, both looking straight at the landmark at
        # (0, 1000): one 50 m short of it, one 400 m short of it.
        catalog = _catalog(["L"], [0.0], [1000.0])
        table = _identity_table("T1", "L")
        east, north, heading = (np.array([0.0, 0.0]),
                                np.array([950.0, 600.0]), np.zeros(2))
        before = pf.pose_log_likelihood(
            east, north, heading, self._meas(None), table, catalog, 0.2)
        capped = pf.pose_log_likelihood(
            east, north, heading, self._meas(100.0), table, catalog, 0.2,
            range_softness=0.25)
        # 50 m range: inside the cap, untouched.
        self.assertAlmostEqual(capped[0], before[0], places=9)
        # 400 m range against a 100 m cap: the landmark branch pays
        # 0.5 * (300/25)^2 = 72 nats and collapses onto the null floor.
        self.assertLess(capped[1], before[1] - 1.0)
        self.assertAlmostEqual(
            capped[1], math.log(0.2) - math.log(2.0 * math.pi), places=6)

    def test_disabled_config_strips_caps_before_any_kernel(self):
        catalog = _catalog(["L"], [0.0], [1000.0])
        tables = {"T1": _identity_table("T1", "L")}
        odometry = [structs.OdometryDelta(
            keyframe_idx=1, forward_m=0.0, left_m=0.0, delta_yaw_cw_rad=0.0,
            sigma_m=1.0, sigma_yaw_rad=0.01)]
        base = dict(n_particles=64, seed=1,
                    init=structs.GaussianInit(0.0, 600.0, 10.0),
                    measurement_backend="numpy",
                    proposal=structs.ProposalConfig(enabled=False),
                    modes=structs.ModeConfig(enabled=False))
        capped = [self._meas(100.0, anchor=1)]
        plain = [self._meas(None, anchor=1)]
        off = pf.run_filter(
            structs.FilterConfig(range_cap_enabled=False, **base),
            catalog, odometry, capped, tables)
        ref = pf.run_filter(
            structs.FilterConfig(range_cap_enabled=False, **base),
            catalog, odometry, plain, tables)
        on = pf.run_filter(
            structs.FilterConfig(range_cap_enabled=True, **base),
            catalog, odometry, capped, tables)
        self.assertEqual(off.particle_history_sha256,
                         ref.particle_history_sha256)
        self.assertNotEqual(on.particle_history_sha256,
                            ref.particle_history_sha256)


if __name__ == "__main__":
    unittest.main()
