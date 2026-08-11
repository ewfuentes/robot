"""End-to-end behaviour on synthetic scenarios (fake tracklets, identity
matcher, made-up trajectories).

Accuracy envelopes with fixed seeds. These are only meaningful because
`consistency_test.py` separately establishes that the filter's reported
uncertainty is honest — an accuracy assertion on an inconsistent filter
measures nothing. Instantiates the design doc's T-F3/F4/F5/F7/F10/F11.
"""

import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    filter as pf,
    scenario,
    structs,
)

# keyframe_period_s=5 -> 25 m steps -> harbor_loop is 241 keyframes.
_PERIOD_S = 5.0


def _run(scenario_config, filter_config):
    data = scenario.generate(scenario_config)
    history = pf.run_filter(filter_config, data.catalog, data.odometry,
                            data.measurements, data.tables)
    return data, history


def _local_init(data, offset_e=300.0, offset_n=-200.0, sigma=500.0):
    start = data.truth[0]
    return structs.GaussianInit(start.east_m + offset_e,
                                start.north_m + offset_n, sigma)


class LocalConvergenceTest(unittest.TestCase):
    def test_converges_and_stays(self):
        cfg = scenario.harbor_loop(keyframe_period_s=_PERIOD_S)
        data = scenario.generate(cfg)
        filter_config = structs.FilterConfig(
            n_particles=5000, seed=5, init=_local_init(data))
        _, history = _run(cfg, filter_config)

        errors = pf.position_errors_m(history.health, data.truth)
        heading_errors = pf.heading_errors_deg(history.health, data.truth)
        last_quarter = slice(3 * len(errors) // 4, None)

        self.assertLess(float(np.median(errors[last_quarter])), 80.0)
        self.assertLess(float(np.max(errors[last_quarter])), 200.0)
        self.assertLess(float(np.median(heading_errors[last_quarter])), 5.0)
        self.assertLess(float(np.median(errors[60:])), 100.0)


class GlobalInitConvergenceTest(unittest.TestCase):
    def test_uniform_box_converges(self):
        """Brute-force global init over 5x5 km + uniform heading. Staggered
        sparse anchors act as annealing. Scored by posterior mass near truth
        rather than mean error: the belief is multimodal by construction
        early on, and the mean of a multimodal cloud describes no hypothesis
        the filter holds (§5.1)."""
        cfg = scenario.harbor_loop(
            keyframe_period_s=_PERIOD_S, epoch_length_keyframes=3,
            bearing_sigma_deg=3.0)
        data = scenario.generate(cfg)
        filter_config = structs.FilterConfig(
            n_particles=120000, seed=3,
            init=structs.UniformBoxInit(-2500.0, 2500.0, -2500.0, 2500.0),
            position_roughening_m=15.0,
            heading_roughening_deg=1.0,
            checkpoint_every=50)
        history = pf.run_filter(filter_config, data.catalog, data.odometry,
                                data.measurements, data.tables)

        final_truth = data.truth[-1]
        mass = pf.mass_within_radius(history.final_belief, final_truth.east_m,
                                     final_truth.north_m, 250.0)
        self.assertGreater(mass, 0.5,
                           f"only {mass:.0%} of the posterior within 250 m "
                           f"of truth")
        map_errors = pf.map_position_errors_m(history.health, data.truth)
        self.assertLess(float(map_errors[-1]), 250.0)


class OdometryOnlyTest(unittest.TestCase):
    def test_uncertainty_grows_without_measurements(self):
        """T-F4a-lite: no tracklets -> belief degrades gracefully to
        odometry-only; position spread grows, mean stays near truth."""
        cfg = scenario.harbor_loop(keyframe_period_s=_PERIOD_S)
        data = scenario.generate(cfg)
        filter_config = structs.FilterConfig(
            n_particles=5000, seed=5,
            init=structs.GaussianInit(data.truth[0].east_m,
                                      data.truth[0].north_m, 1.0))
        history = pf.run_filter(filter_config, data.catalog, data.odometry,
                                [], {})

        errors = pf.position_errors_m(history.health, data.truth)
        std_start = history.health[1].position_std_m
        std_end = history.health[-1].position_std_m
        self.assertGreater(std_end, 2.0 * std_start)
        self.assertLess(float(errors[-1]), 50.0)


class ClutterRobustnessTest(unittest.TestCase):
    """T-F4: the null hypothesis must absorb tracklets with no catalog
    counterpart, at the SHIPPED defaults — a safety property that is only
    meaningful if it holds for the configuration people actually run."""

    # With every tracklet clutter, position is essentially unconstrained, so
    # the final *error* is close to a coin flip — the property that separates
    # a working null from a starved one is whether the filter stays
    # appropriately UNCERTAIN, so score error/sigma over several seeds.
    _SEEDS = (5, 6, 7)

    def _clutter_runs(self, pi0):
        cfg = scenario.harbor_loop(keyframe_period_s=_PERIOD_S,
                                   clutter_only=True)
        data = scenario.generate(cfg)
        histories = []
        for seed in self._SEEDS:
            filter_config = structs.FilterConfig(
                n_particles=3000, seed=seed, init=_local_init(data), pi0=pi0)
            histories.append(pf.run_filter(filter_config, data.catalog,
                                           data.odometry, data.measurements,
                                           data.tables))
        errors = [float(pf.position_errors_m(h.health, data.truth)[-1])
                  for h in histories]
        sigmas = [h.health[-1].position_std_m for h in histories]
        return histories, np.array(errors), np.array(sigmas)

    @staticmethod
    def _default_pi0():
        return structs.FilterConfig(
            n_particles=1, seed=0,
            init=structs.GaussianInit(0.0, 0.0, 1.0)).pi0

    def test_no_confident_wrong_fix_at_defaults(self):
        histories, errors, sigmas = self._clutter_runs(self._default_pi0())

        null_shares = [assoc.null_share
                       for history in histories
                       for record in history.health
                       for assoc in record.associations]
        self.assertGreater(len(null_shares), 50)
        self.assertGreater(float(np.mean(null_shares)), 0.15)

        for error, sigma in zip(errors, sigmas):
            self.assertFalse(sigma < 100.0 and error > 300.0,
                             f"confident wrong fix: std={sigma:.0f} m, "
                             f"error={error:.0f} m")

    def test_null_starvation_ablation_is_worse(self):
        """T-F4b: the ablation must demonstrate the *contrast* the null
        hypothesis buys, not pin an absolute failure magnitude — otherwise
        improving clutter robustness breaks the test."""
        _, default_errors, default_sigmas = self._clutter_runs(
            self._default_pi0())
        _, starved_errors, starved_sigmas = self._clutter_runs(pi0=1e-4)

        default_ratio = float(np.mean(default_errors / default_sigmas))
        starved_ratio = float(np.mean(starved_errors / starved_sigmas))
        self.assertGreater(starved_ratio, 3.0 * default_ratio,
                           f"null starvation should leave the filter far "
                           f"more overconfident: error/sigma "
                           f"{starved_ratio:.1f} starved vs "
                           f"{default_ratio:.1f} at defaults")
        self.assertLess(float(np.mean(starved_sigmas)),
                        float(np.mean(default_sigmas)))


class MultimodalityPreservationTest(unittest.TestCase):
    """T-F3: a symmetric world produces a genuinely bimodal posterior. The
    parent document calls premature unimodality unrecoverable by design, so
    this is the failure this test exists to catch."""

    def test_symmetric_landmarks_keep_both_modes(self):
        cfg = scenario.symmetric_pair(keyframe_period_s=_PERIOD_S)
        data = scenario.generate(cfg)
        # The two landmarks are the same type, so the matcher cannot break
        # the tie: every tracklet scores both equally.
        both = [structs.CompatibilityEntry(lm_id, cfg.identity_clip)
                for lm_id in data.landmark_ids]
        tables = {
            tid: structs.CompatibilityTable(
                tracklet_id=table.tracklet_id,
                matcher_version=table.matcher_version, entries=both,
                default_log_lr=table.default_log_lr, clip_lo=table.clip_lo,
                clip_hi=table.clip_hi, status=table.status)
            for tid, table in data.tables.items()}

        # Init symmetric about the trajectory (which runs up the east=0 axis).
        filter_config = structs.FilterConfig(
            n_particles=60000, seed=1,
            init=structs.UniformBoxInit(-1500.0, 1500.0, -1200.0, 1200.0),
            checkpoint_every=1000)
        history = pf.run_filter(filter_config, data.catalog, data.odometry,
                                data.measurements, tables)

        truth = data.truth[-1]
        belief = history.final_belief
        east = belief.east_m
        weights = belief.normalized_weights()
        # Mirror ambiguity is about the east=0 axis: mass should survive on
        # both sides rather than collapsing to one.
        east_side = float(weights[east > 0.0].sum())
        west_side = float(weights[east < 0.0].sum())
        self.assertGreater(min(east_side, west_side), 0.15,
                           f"belief collapsed to one mode: east={east_side:.2f}"
                           f" west={west_side:.2f}")
        # ...and the true pose must remain one of the surviving modes.
        self.assertGreater(
            pf.mass_within_radius(belief, truth.east_m, truth.north_m, 400.0),
            0.1)


class LlrSaturationTest(unittest.TestCase):
    """T-F5: an adversarial table at the clip bound on a geometrically wrong
    landmark must not steamroll the geometric term."""

    def test_geometry_wins_over_saturated_llr(self):
        cfg = scenario.harbor_loop(keyframe_period_s=_PERIOD_S)
        data = scenario.generate(cfg)
        # Point one tracklet's whole LLR mass at the wrong landmark.
        victim = data.landmark_ids[0]
        liar = data.landmark_ids[1]
        tables = dict(data.tables)
        original = tables[f"trk_{victim}"]
        tables[f"trk_{victim}"] = structs.CompatibilityTable(
            tracklet_id=original.tracklet_id,
            matcher_version="adversarial",
            entries=[structs.CompatibilityEntry(liar, original.clip_hi),
                     structs.CompatibilityEntry(victim, original.clip_lo)],
            default_log_lr=original.clip_lo, clip_lo=original.clip_lo,
            clip_hi=original.clip_hi, status="fast")

        filter_config = structs.FilterConfig(
            n_particles=20000, seed=5, init=_local_init(data))
        history = pf.run_filter(filter_config, data.catalog, data.odometry,
                                data.measurements, tables)
        errors = pf.position_errors_m(history.health, data.truth)
        self.assertLess(float(np.median(errors[-40:])), 250.0,
                        "a saturated LLR on the wrong landmark defeated the "
                        "geometric term")


class MapErrorRobustnessTest(unittest.TestCase):
    """T-F10: catalog position error is absorbed by kappa_eff; accuracy
    degrades smoothly rather than falling off a cliff."""

    def test_degrades_smoothly(self):
        errors = {}
        for sigma_m in (0.0, 10.0, 50.0):
            cfg = scenario.harbor_loop(keyframe_period_s=_PERIOD_S,
                                       catalog_position_sigma_m=sigma_m)
            data = scenario.generate(cfg)
            filter_config = structs.FilterConfig(
                n_particles=20000, seed=5, init=_local_init(data))
            history = pf.run_filter(filter_config, data.catalog,
                                    data.odometry, data.measurements,
                                    data.tables)
            final = data.truth[-1]
            errors[sigma_m] = math.hypot(
                history.health[-1].mean_east_m - final.east_m,
                history.health[-1].mean_north_m - final.north_m)
        self.assertLess(errors[10.0], 200.0, f"{errors}")
        self.assertLess(errors[50.0], 400.0, f"{errors}")


class ParticleCountInvarianceTest(unittest.TestCase):
    """T-F11: results must be statistically stable in particle count, not
    quietly dependent on it."""

    def test_estimates_agree_across_counts(self):
        cfg = scenario.harbor_loop(keyframe_period_s=_PERIOD_S)
        data = scenario.generate(cfg)
        estimates = []
        for n_particles in (10000, 40000):
            filter_config = structs.FilterConfig(
                n_particles=n_particles, seed=5, init=_local_init(data))
            history = pf.run_filter(filter_config, data.catalog,
                                    data.odometry, data.measurements,
                                    data.tables)
            estimates.append(pf.mean_pose(history.final_belief))
        spread = math.hypot(estimates[0][0] - estimates[1][0],
                            estimates[0][1] - estimates[1][1])
        self.assertLess(spread, 120.0,
                        f"estimate moved {spread:.0f} m with particle count")


class DeterminismTest(unittest.TestCase):
    def test_bit_identical_replay(self):
        """T-U7 at E2E level: same (config, seed, inputs) -> same hash."""
        cfg = scenario.harbor_loop(keyframe_period_s=_PERIOD_S)
        data = scenario.generate(cfg)
        filter_config = structs.FilterConfig(
            n_particles=2000, seed=5, init=_local_init(data))
        _, h1 = _run(cfg, filter_config)
        _, h2 = _run(cfg, filter_config)
        self.assertEqual(h1.particle_history_sha256,
                         h2.particle_history_sha256)


class MeasurementOrderInvarianceTest(unittest.TestCase):
    def test_within_keyframe_permutation(self):
        """T-F7-lite: permuting measurement order within keyframes changes
        only fp rounding, not the posterior (weights add commutatively)."""
        cfg = scenario.harbor_loop(keyframe_period_s=_PERIOD_S,
                                   epoch_length_keyframes=1)
        data = scenario.generate(cfg)
        filter_config = structs.FilterConfig(
            n_particles=2000, seed=5, init=_local_init(data))
        h_forward = pf.run_filter(filter_config, data.catalog, data.odometry,
                                  data.measurements, data.tables)
        h_reversed = pf.run_filter(filter_config, data.catalog, data.odometry,
                                   list(reversed(data.measurements)),
                                   data.tables)

        final_fwd = h_forward.health[-1]
        final_rev = h_reversed.health[-1]
        self.assertAlmostEqual(final_fwd.mean_east_m, final_rev.mean_east_m,
                               delta=1e-6)
        self.assertAlmostEqual(final_fwd.mean_north_m,
                               final_rev.mean_north_m, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
