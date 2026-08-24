"""Mixture proposal: hypothesis generation, provenance, and recovery (§5.5).

Design doc T-F6. The tests are deliberately biased toward *contrast*
assertions ("recovery converges where brute force does not", "a healthy run
is unchanged") rather than absolute thresholds, because the value of the
proposal is relative to not having one, and absolute numbers would pin in
whatever the implementation happens to do today.

The most important test here is RecoveryConsistencyTest: injection weighting
is the statistically delicate part of this feature, and a filter that
recovers to the right place while badly overconfident is not recovered. NEES
is the only assertion that can see the difference.
"""

import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    filter as pf,
    metrics,
    proposal,
    resection,
    scenario,
    structs,
)

_PERIOD_S = 5.0


def _propose(measurements, tables, catalog, config, event_id, keyframe_idx,
             trigger, *, particle_budget=20000):
    return proposal.propose(
        measurements, tables, catalog, config, event_id, keyframe_idx,
        trigger, particle_budget=particle_budget)


def _proposal_result(hypotheses, particle_budget):
    return proposal.ProposalResult(
        event_id=0, keyframe_idx=0, trigger="test",
        hypotheses=hypotheses, particle_budget=particle_budget,
        n_tracklets_considered=1,
        n_combinations_total=len(hypotheses),
        n_combinations_enumerated=len(hypotheses),
        n_combinations_sampled=0, n_combinations_geometry_pruned=0,
        n_partially_represented_ties=0, n_solution_clusters_merged=0,
        represented_compatibility_mass=1.0)


def _kidnapped_scenario(jump_east_m=1500.0, jump_north_m=-1200.0,
                        jump_at=120, **overrides):
    """A run where the vehicle teleports mid-trajectory (scenario.apply_kidnap
    does the work; kept as a helper so the tests read as T-F6)."""
    data = scenario.generate(scenario.harbor_loop(
        max_visible_range_m=10000.0, keyframe_period_s=_PERIOD_S, **overrides))
    kidnapped = scenario.apply_kidnap(data, jump_at, jump_east_m,
                                      jump_north_m)
    return kidnapped, kidnapped.truth, kidnapped.measurements


class HypothesisGenerationTest(unittest.TestCase):
    def setUp(self):
        self.data = scenario.generate(
            scenario.harbor_loop(max_visible_range_m=10000.0, keyframe_period_s=_PERIOD_S,
                                 epoch_length_keyframes=1,
                                 bearing_sigma_deg=0.5))
        self.config = structs.ProposalConfig()

    def _measurements_at(self, keyframe_idx):
        return [m for m in self.data.measurements
                if m.anchor_keyframe_idx == keyframe_idx]

    def _window(self, keyframe_idx=90):
        """Staggered epochs mean one keyframe rarely carries two bearings,
        so the proposal consumes a short window (see proposal.py)."""
        start = keyframe_idx - self.config.window_keyframes
        return [m for m in self.data.measurements
                if start <= m.anchor_keyframe_idx <= keyframe_idx]

    def test_truth_is_among_the_hypotheses(self):
        """Recall: if the true pose is never proposed, nothing downstream
        can recover it."""
        truth_by_kf = {t.keyframe_idx: t for t in self.data.truth}
        hits = 0
        tried = 0
        for keyframe_idx in (10, 40, 90, 150, 200):
            measurements = self._window(keyframe_idx)
            if len(measurements) < 3:
                continue
            tried += 1
            result = _propose(measurements, self.data.tables,
                                      self.data.catalog, self.config,
                                      event_id=0, keyframe_idx=keyframe_idx,
                                      trigger="test")
            sampled = proposal.sample_particles(
                result, 20000, self.config, np.random.default_rng(0))
            pose = truth_by_kf[keyframe_idx]
            # Recall is measured on the SAMPLED cloud, not on hypothesis
            # centres: arcs and discs have no centre, and what matters is
            # whether injected particles land near truth.
            near = np.hypot(sampled[0] - pose.east_m,
                            sampled[1] - pose.north_m) < 250.0
            if bool(np.any(near)):
                hits += 1
        self.assertGreater(tried, 3)
        self.assertEqual(hits, tried, "truth missing from the hypothesis set")

    def test_provenance_reproduces_the_pose(self):
        """`[CONTRACT]` provenance must be *correct*, not merely present:
        every sampled particle has to satisfy the bearings its recorded
        tracklets and landmarks imply. Checking that a field is populated
        would pass even if the ids were shuffled."""
        measurements = self._window()
        by_id = {m.tracklet_id: m for m in measurements}
        result = _propose(measurements, self.data.tables,
                                  self.data.catalog, self.config, 0, 90,
                                  "test")
        self.assertGreater(len(result.hypotheses), 0)
        strict = structs.ProposalConfig(injection_sigma_m=0.0,
                                        injection_heading_sigma_deg=0.0)
        rng = np.random.default_rng(0)
        for hypothesis in result.hypotheses:
            self.assertEqual(len(set(hypothesis.landmark_ids)),
                             len(hypothesis.landmark_ids))
            east, north, heading = hypothesis.sample(24, strict, rng)
            self.assertGreater(east.size, 0)
            for tracklet_id, landmark_id in zip(hypothesis.tracklet_ids,
                                                hypothesis.landmark_ids):
                index = self.data.catalog.index_of(landmark_id)
                predicted = geo.wrap_rad(
                    geo.compass_bearing_rad(
                        self.data.catalog.east_m[index] - east,
                        self.data.catalog.north_m[index] - north) - heading)
                observed = math.radians(by_id[tracklet_id].bearing_forward_cw_deg)
                residual = np.abs(geo.wrap_rad(predicted - observed))
                self.assertLess(
                    float(np.max(residual)), math.radians(6.0),
                    f"{hypothesis.kind} hypothesis does not satisfy the "
                    f"bearing it claims to come from")

    def test_all_three_hypothesis_kinds_are_generated(self):
        """One landmark pins heading, two give an arc, three give a fix —
        each is a usable proposal and dropping the weaker kinds throws away
        the cases where only one or two landmarks are visible."""
        result = _propose(self._window(), self.data.tables,
                                  self.data.catalog, self.config, 0, 90,
                                  "test")
        counts = result.counts_by_kind()
        for kind in (proposal.TRIPLE, proposal.PAIR, proposal.SINGLE):
            self.assertGreater(counts.get(kind, 0), 0, f"no {kind} generated")

    def test_hypotheses_are_ranked_by_residual_within_kind(self):
        result = _propose(self._window(), self.data.tables,
                                  self.data.catalog, self.config, 0, 90,
                                  "test")
        for kind in (proposal.TRIPLE, proposal.PAIR, proposal.SINGLE):
            residuals = [h.residual_rad for h in result.hypotheses
                         if h.kind == kind]
            self.assertEqual(residuals, sorted(residuals))

    @staticmethod
    def _tied_fixture(n_landmarks):
        class Catalog:
            def __init__(self, count):
                self.landmark_ids = [f"lm_{index:03d}" for index in range(count)]
                angles = np.linspace(0.0, 2.0 * math.pi, count,
                                     endpoint=False)
                self.east_m = 3000.0 * np.sin(angles)
                self.north_m = 3000.0 * np.cos(angles)
                self.max_visible_range_m = np.full(count, 10000.0)
                self._index = {value: index for index, value
                               in enumerate(self.landmark_ids)}

            def __contains__(self, landmark_id):
                return landmark_id in self._index

            def index_of(self, landmark_id):
                return self._index[landmark_id]

        catalog = Catalog(n_landmarks)
        measurement = structs.TrackletMeasurement("trk", 0, 12.0, 100.0)
        table = structs.CompatibilityTable(
            tracklet_id="trk", matcher_version="tied",
            entries=[structs.CompatibilityEntry(landmark_id, 2.0)
                     for landmark_id in catalog.landmark_ids],
            default_log_lr=-2.0, clip_lo=-4.0, clip_hi=4.0,
            status="fast")
        return catalog, measurement, table

    def test_small_candidate_space_is_enumerated_without_truncation(self):
        catalog, measurement, table = self._tied_fixture(4)
        config = structs.ProposalConfig(exhaustive_tuple_limit=8)
        result = _propose(
            [measurement], {"trk": table}, catalog, config, 0, 0, "test",
            particle_budget=4 * config.min_particles_single)
        self.assertEqual(result.n_combinations_total, 4)
        self.assertEqual(result.n_combinations_enumerated, 4)
        self.assertEqual(result.n_combinations_sampled, 0)
        self.assertEqual(len(result.hypotheses), 4)
        self.assertAlmostEqual(result.represented_compatibility_mass, 1.0)

    def test_large_tie_is_systematically_sampled_and_reported(self):
        catalog, measurement, table = self._tied_fixture(20)
        config = structs.ProposalConfig(
            exhaustive_tuple_limit=4,
            tuple_samples_per_active_solution=4)
        result = _propose(
            [measurement], {"trk": table}, catalog, config, 0, 0, "test",
            particle_budget=config.min_particles_single)
        reversed_table = structs.CompatibilityTable(
            tracklet_id=table.tracklet_id,
            matcher_version=table.matcher_version,
            entries=list(reversed(table.entries)),
            default_log_lr=table.default_log_lr, clip_lo=table.clip_lo,
            clip_hi=table.clip_hi, status=table.status)
        reordered = _propose(
            [measurement], {"trk": reversed_table}, catalog, config,
            0, 0, "test", particle_budget=config.min_particles_single)

        self.assertEqual(result.n_combinations_total, 20)
        self.assertEqual(result.n_combinations_sampled, 4)
        self.assertEqual(result.n_combinations_skipped, 16)
        self.assertEqual(result.n_partially_represented_ties, 1)
        self.assertEqual(
            [item.landmark_ids for item in result.hypotheses],
            [item.landmark_ids for item in reordered.hypotheses])

    def test_near_duplicate_pose_solutions_share_one_budget(self):
        config = structs.ProposalConfig()
        hypotheses = [
            proposal.PointHypothesis(
                kind=proposal.TRIPLE, tracklet_ids=("a", "b", "c"),
                landmark_ids=("x", "y", "z"), east_m=100.0,
                north_m=200.0, heading_rad=0.2,
                compatibility_mass=0.5),
            proposal.PointHypothesis(
                kind=proposal.TRIPLE, tracklet_ids=("d", "e", "f"),
                landmark_ids=("u", "v", "w"), east_m=110.0,
                north_m=205.0, heading_rad=0.21,
                compatibility_mass=0.4),
            proposal.PointHypothesis(
                kind=proposal.TRIPLE, tracklet_ids=("g", "h", "i"),
                landmark_ids=("r", "s", "t"), east_m=1500.0,
                north_m=-800.0, heading_rad=-1.0,
                compatibility_mass=0.1),
        ]
        clustered, merged = proposal._cluster_hypotheses(hypotheses, config)
        self.assertEqual(len(clustered), 2)
        self.assertEqual(merged, 1)

    def test_coarse_bearings_widen_the_fix_rather_than_being_rejected(self):
        """What a coarse bearing implies is an IMPRECISE fix, not an
        untrustworthy one. Measured on a 3-landmark fixture, the true-identity
        solution's own residual grows with noise (0.0/4.4/20.5/33.9 deg at
        sigma 1/5/15/25), so gating below that throws away real solutions —
        the honest response is to inject them with the spread the geometry
        actually supports."""
        sharp = self._window()
        coarse = [structs.TrackletMeasurement(m.tracklet_id,
                                              m.anchor_keyframe_idx,
                                              m.bearing_forward_cw_deg, 20.0)
                  for m in sharp]
        config = structs.ProposalConfig()

        def triples(window):
            result = _propose(window, self.data.tables,
                                      self.data.catalog, config, 0, 90,
                                      "test")
            return [h for h in result.hypotheses if h.kind == proposal.TRIPLE]

        sharp_fixes, coarse_fixes = triples(sharp), triples(coarse)
        self.assertGreater(len(coarse_fixes), 0,
                           "coarse bearings produced no fix at all — the "
                           "residual gate is rejecting true solutions")
        sharp_spread = min(h.position_sigma_m for h in sharp_fixes)
        coarse_spread = min(h.position_sigma_m for h in coarse_fixes)
        self.assertGreater(coarse_spread, 5.0 * sharp_spread,
                           f"a fix from sigma=13deg bearings was injected as "
                           f"tightly as one from sigma=1deg: "
                           f"{coarse_spread:.0f} m vs {sharp_spread:.0f} m")

        # ...and the widened spread is what actually reaches the particles.
        rng = np.random.default_rng(0)
        east, _, _ = coarse_fixes[0].sample(4000, config, rng)
        self.assertGreater(float(np.std(east)),
                           2.0 * config.injection_sigma_m)

    def test_injection_spread_is_floored_and_capped(self):
        config = structs.ProposalConfig(injection_sigma_m=80.0,
                                        max_injection_sigma_m=500.0)
        rng = np.random.default_rng(1)
        tight = proposal.PointHypothesis(
            kind=proposal.TRIPLE, tracklet_ids=("a",), landmark_ids=("x",),
            position_sigma_m=1.0)
        huge = proposal.PointHypothesis(
            kind=proposal.TRIPLE, tracklet_ids=("a",), landmark_ids=("x",),
            position_sigma_m=99000.0)
        self.assertAlmostEqual(float(np.std(tight.sample(20000, config,
                                                         rng)[0])),
                               80.0, delta=6.0)
        self.assertAlmostEqual(float(np.std(huge.sample(20000, config,
                                                        rng)[0])),
                               500.0, delta=30.0)

    def test_degrades_by_kind_as_tracklets_are_removed(self):
        """Fewer landmarks should cost sharpness, not the whole proposal."""
        window = self._window()
        by_tracklet = {}
        for meas in window:
            by_tracklet.setdefault(meas.tracklet_id, meas)
        ordered = list(by_tracklet.values())

        kinds_for = lambda subset: set(_propose(
            subset, self.data.tables, self.data.catalog, self.config, 0, 90,
            "test").counts_by_kind())
        self.assertIn(proposal.TRIPLE, kinds_for(ordered[:3]))
        self.assertEqual(kinds_for(ordered[:2]),
                         {proposal.PAIR, proposal.SINGLE})
        self.assertEqual(kinds_for(ordered[:1]), {proposal.SINGLE})
        self.assertEqual(kinds_for([]), set())

    def test_single_landmark_collapses_heading_but_not_position(self):
        """The single-landmark case is worth having precisely because it
        removes the heading dimension exactly."""
        window = self._window()[:1]
        result = _propose(window, self.data.tables,
                                  self.data.catalog, self.config, 0, 90,
                                  "test")
        strict = structs.ProposalConfig(injection_sigma_m=0.0,
                                        injection_heading_sigma_deg=0.0)
        east, north, heading, _ = proposal.sample_particles(
            result, 4000, strict, np.random.default_rng(0))
        self.assertGreater(east.size, 0)
        # Position spreads over the visibility disc...
        self.assertGreater(float(np.std(east)), 1000.0)
        # ...but heading is determined by position, not free: recomputing it
        # from the landmark reproduces what was sampled.
        index = self.data.catalog.index_of(
            result.hypotheses[0].landmark_ids[0])
        implied = geo.wrap_rad(
            geo.compass_bearing_rad(
                self.data.catalog.east_m[index] - east,
                self.data.catalog.north_m[index] - north)
            - math.radians(window[0].bearing_forward_cw_deg))
        self.assertLess(
            float(np.max(np.abs(geo.wrap_rad(implied - heading)))),
            math.radians(1.0))

    def test_area_uniform_sampling_of_the_visibility_disc(self):
        """Sampling uniformly in range instead of area piles particles up
        near the landmark and silently biases the proposal density."""
        window = self._window()[:1]
        result = _propose(window, self.data.tables,
                                  self.data.catalog, self.config, 0, 90,
                                  "test")
        hypothesis = next(h for h in result.hypotheses
                          if h.kind == proposal.SINGLE)
        east, north, _ = hypothesis.sample(40000, self.config,
                                           np.random.default_rng(1))
        radius = np.hypot(east - hypothesis.landmark[0],
                          north - hypothesis.landmark[1])
        # Uniform in area => half the mass beyond R/sqrt(2).
        inner = float(np.mean(radius < hypothesis.max_range_m / math.sqrt(2)))
        self.assertAlmostEqual(inner, 0.5, delta=0.02)

    def test_window_approximation_error_grows_with_travel(self):
        """Bearings across the window are treated as simultaneous, which is
        wrong by translation/range. Tested directly rather than through the
        filter: take each bearing from where the vehicle actually was that
        many keyframes earlier, resect as if they were simultaneous, and
        watch the fix walk away from truth as the spacing grows."""
        truth_by_kf = {t.keyframe_idx: t for t in self.data.truth}
        anchor = truth_by_kf[120]
        landmark_ids = self.data.landmark_ids[:3]
        positions = [(float(self.data.catalog.east_m[
                          self.data.catalog.index_of(lm)]),
                      float(self.data.catalog.north_m[
                          self.data.catalog.index_of(lm)]))
                     for lm in landmark_ids]

        errors = {}
        for spacing in (0, 1, 4, 20):
            bearings = []
            for offset, (east, north) in enumerate(positions):
                # Bearing i was taken `offset * spacing` keyframes ago.
                pose = truth_by_kf[120 - offset * spacing]
                bearings.append(float(geo.wrap_rad(
                    geo.compass_bearing_rad(east - pose.east_m,
                                                north - pose.north_m)
                    - math.radians(anchor.course_world_cw_deg))))
            solutions = resection.resect_three(
                positions, bearings,
                residual_tolerance_rad=math.radians(20.0))
            errors[spacing] = min(
                (math.hypot(s.east_m - anchor.east_m,
                            s.north_m - anchor.north_m)
                 for s in solutions), default=float("inf"))
        self.assertLess(errors[0], 1.0, f"exact case is wrong: {errors}")
        self.assertLess(errors[self.config.window_keyframes], 250.0,
                        f"the window in use is already too wide: {errors}")
        self.assertGreater(errors[20], errors[4],
                           f"error should grow with travel: {errors}")

    def test_uninformative_tables_yield_nothing(self):
        """Proposal starvation: with a flat matcher there is no landmark
        identity to resect against, and the proposal must say so rather than
        invent one."""
        flat = {tid: structs.CompatibilityTable(
            tracklet_id=tid, matcher_version="flat", entries=[],
            default_log_lr=0.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
            for tid in self.data.tables}
        result = _propose(self._window(), flat,
                                  self.data.catalog, self.config, 0, 90,
                                  "test")
        self.assertEqual(result.hypotheses, [])
        self.assertEqual(result.n_tracklets_considered, 0)

    def test_sampling_is_deterministic(self):
        result = _propose(self._window(),
                                  self.data.tables, self.data.catalog,
                                  self.config, 0, 90, "test")
        draws = [proposal.sample_particles(result, 500, self.config,
                                           np.random.default_rng(4))
                 for _ in range(2)]
        np.testing.assert_array_equal(draws[0][0], draws[1][0])
        np.testing.assert_array_equal(draws[0][3], draws[1][3])

    def test_sampling_returns_exact_count_and_honors_point_floors(self):
        hypotheses = [proposal.PointHypothesis(
            kind=proposal.TRIPLE, tracklet_ids=(f"t{index}",),
            landmark_ids=(f"l{index}",), east_m=1000.0 * index,
            compatibility_mass=1.0 / 3.0)
            for index in range(3)]
        config = structs.ProposalConfig(
            injection_sigma_m=0.0, injection_heading_sigma_deg=0.0)
        result = _proposal_result(hypotheses, 96)
        sampled = proposal.sample_particles(
            result, 96, config, np.random.default_rng(3))
        self.assertEqual(sampled[0].size, 96)
        np.testing.assert_array_equal(
            np.bincount(sampled[3], minlength=3), [32, 32, 32])

        one = proposal.sample_particles(
            _proposal_result(hypotheses[:1], 1), 1, config,
            np.random.default_rng(3))
        self.assertEqual(one[0].size, 1)

    def test_arc_length_guides_remaining_particle_allocation(self):
        arc = resection.arcs_for_signed_angle(
            -1000.0, 0.0, 1000.0, 0.0, math.radians(60.0))[0]
        hypotheses = [proposal.ArcHypothesis(
            kind=proposal.PAIR, tracklet_ids=(f"a{index}", f"b{index}"),
            landmark_ids=(f"x{index}", f"y{index}"), arc=arc,
            landmark_a=(-1000.0, 0.0), landmark_b=(1000.0, 0.0),
            bearing_a_rad=0.0, arc_length_m=length,
            compatibility_mass=0.5)
            for index, length in enumerate((100.0, 10000.0))]
        config = structs.ProposalConfig(
            injection_sigma_m=0.0, injection_heading_sigma_deg=0.0)
        sampled = proposal.sample_particles(
            _proposal_result(hypotheses, 256), 256, config,
            np.random.default_rng(5))
        counts = np.bincount(sampled[3], minlength=2)
        self.assertGreaterEqual(int(counts.min()), config.min_particles_arc)
        self.assertGreater(counts[1], counts[0])


class InitTriggerTest(unittest.TestCase):
    def test_fires_at_the_first_keyframe_with_bearings(self):
        """Real exports start observing several keyframes in; keying the
        initial proposal to keyframe 0 means it silently never fires and the
        prior is left to brute force."""
        data = scenario.generate(scenario.harbor_loop(
            max_visible_range_m=10000.0, keyframe_period_s=_PERIOD_S, epoch_length_keyframes=1))
        delayed = [structs.TrackletMeasurement(
            m.tracklet_id, m.anchor_keyframe_idx, m.bearing_forward_cw_deg, m.kappa)
            for m in data.measurements if m.anchor_keyframe_idx >= 7]
        config = structs.FilterConfig(
            n_particles=4000, seed=1,
            init=structs.UniformBoxInit(-3000.0, 3000.0, -3000.0, 3000.0),
            checkpoint_every=1000)
        history = pf.run_filter(config, data.catalog, data.odometry,
                                delayed, data.tables)
        init_events = [e for e in history.proposal_events
                       if e.trigger == "init"]
        self.assertEqual(len(init_events), 1,
                         "initial proposal did not fire on a run whose "
                         "bearings start after keyframe 0")
        self.assertEqual(init_events[0].keyframe_idx, 7)
        self.assertGreater(init_events[0].n_injected, 0)


class KidnappedRecoveryTest(unittest.TestCase):
    """T-F6."""

    def _run(self, data, measurements, **proposal_overrides):
        config = structs.FilterConfig(
            n_particles=20000, seed=5,
            init=structs.GaussianInit(data.truth[0].east_m,
                                      data.truth[0].north_m, 300.0),
            checkpoint_every=1000,
            proposal=structs.ProposalConfig(**proposal_overrides))
        return pf.run_filter(config, data.catalog, data.odometry,
                             measurements, data.tables)

    def test_recovers_after_a_jump(self):
        data, truth, measurements = _kidnapped_scenario()
        history = self._run(data, measurements)
        errors = metrics.map_position_errors_m(history.health, truth)

        self.assertGreater(float(np.max(errors[120:135])), 500.0,
                           "the kidnap did not actually displace the belief")
        self.assertLess(float(np.median(errors[-30:])), 300.0,
                        f"never recovered; final errors "
                        f"{np.round(errors[-5:], 0)}")
        recovered = np.nonzero(errors[120:] < 300.0)[0]
        self.assertGreater(len(recovered), 0)
        # 60 keyframes before `evidence_gate_selection_charge` was turned on by
        # default; 70 after, which is exactly one `refractory_keyframes` cycle
        # (10). The charge refuses the first post-kidnap event and the next one
        # carries it, so recovery costs one extra cycle. That is the deliberate
        # price of the charge: on real data it stops a marginal late injection
        # destroying a converged belief (boston_harbor_leg1 on good tracks, final
        # 20132 m -> 368 m at an unchanged 79 m median). The bound is 75 rather
        # than 70 so ordinary jitter does not trip it, and it is still tight
        # enough that a *second* lost cycle would.
        self.assertLess(int(recovered[0]), 75,
                        f"recovery took {int(recovered[0])} keyframes; the "
                        f"selection charge costs one refractory cycle, not two")
        self.assertTrue(any(e.trigger in ("null_share", "ess_floor")
                            for e in history.proposal_events),
                        "recovery happened without a proposal firing")

    def test_no_recovery_without_the_proposal(self):
        """Contrast: the same run with the proposal disabled must NOT
        recover, or the test above is not measuring the proposal."""
        data, truth, measurements = _kidnapped_scenario()
        history = self._run(data, measurements, enabled=False)
        errors = metrics.map_position_errors_m(history.health, truth)
        self.assertGreater(float(np.median(errors[-30:])), 400.0,
                           "recovered without the proposal: the kidnap "
                           "scenario is too easy to be a test of it")

    def test_recovers_from_a_single_visible_landmark(self):
        """The other half of T-F6, made a stronger guarantee by the
        single-landmark hypothesis: with only one tracklet visible the
        proposal cannot fix a position outright, but it pins heading and
        spreads position over the visibility disc, which subsequent bearings
        then sharpen. Before the hierarchy existed this case injected
        nothing and the filter stayed confidently wrong."""
        data, truth, measurements = _kidnapped_scenario()
        only_one = [m for m in measurements
                    if m.tracklet_id == measurements[0].tracklet_id]
        history = self._run(data, only_one)

        fired = [e for e in history.proposal_events if e.n_injected]
        self.assertGreater(len(fired), 0,
                           "no proposal fired with one tracklet visible")
        errors = metrics.map_position_errors_m(history.health, truth)
        final_std = history.health[-1].position_std_m
        self.assertFalse(final_std < 150.0 and float(errors[-1]) > 600.0,
                         f"confident wrong fix while starved: "
                         f"std={final_std:.0f} m err={errors[-1]:.0f} m")

    def test_no_hypotheses_when_nothing_is_visible(self):
        """Proposal starvation proper: no usable tracklet, no invention."""
        data, truth, measurements = _kidnapped_scenario()
        history = self._run(data, [])
        self.assertEqual(history.proposal_events, [])


class RecoveryConsistencyTest(unittest.TestCase):
    """The test that guards the injection-weighting approximation.

    `inject_proposal` treats the proposal density as roughly proportional to
    the likelihood instead of computing prior(x)/q(x). If that bias is
    material, the post-recovery belief is confident about the wrong spread,
    and NEES is what shows it.
    """

    def test_nees_in_bounds_when_recovery_succeeds(self):
        """Recovery is not guaranteed on every seed, and the assertion is
        written to say so rather than to hide it.

        Measured on the harbour kidnap: 6 of 8 seeds recover. The two that
        do not have latched onto a *self-consistent wrong* hypothesis — the
        mirror ambiguity a 3-landmark world genuinely admits — and stop
        re-triggering because the bearings then look explained. Holding both
        hypotheses instead of committing is the mode tracker's job (§5.6,
        deferred), so what this test pins is: recovery succeeds for a clear
        majority, and where it succeeds the belief is honest about itself.
        """
        data, truth, measurements = _kidnapped_scenario()
        recovered, neeses = [], []
        for seed in range(8):
            config = structs.FilterConfig(
                n_particles=20000, seed=seed,
                init=structs.GaussianInit(data.truth[0].east_m,
                                          data.truth[0].north_m, 300.0),
                checkpoint_every=1000)
            history = pf.run_filter(config, data.catalog, data.odometry,
                                    measurements, data.tables)
            final = truth[-1]
            error = float(np.median(
                metrics.map_position_errors_m(history.health, truth)[-30:]))
            if error < 300.0:
                recovered.append(seed)
                neeses.append(metrics.position_nees(history.final_belief,
                                               final.east_m, final.north_m))
        self.assertGreaterEqual(len(recovered), 5,
                                f"only {len(recovered)}/8 seeds recovered")
        mean_nees = float(np.mean(neeses))
        self.assertLess(mean_nees, 12.0,
                        f"overconfident after recovery: mean NEES "
                        f"{mean_nees:.1f} over seeds {recovered}")
        self.assertGreater(mean_nees, 0.02,
                           f"implausibly conservative after recovery: "
                           f"{mean_nees:.3f}")


class GlobalInitEfficiencyTest(unittest.TestCase):
    """The claim of §5.5: particle count scales with the number of plausible
    hypotheses, not with search area."""

    def _time_to_converge(self, n_particles, use_proposal, radius_m=300.0):
        """Keyframes until the MAP estimate is within `radius_m` and stays
        there. Time-to-converge, not final error, is the honest metric: given
        240 keyframes a regularized cloud can eventually diffuse to the right
        answer from almost anywhere, so final error hides the difference the
        proposal actually makes."""
        data = scenario.generate(scenario.harbor_loop(
            max_visible_range_m=10000.0, keyframe_period_s=_PERIOD_S, epoch_length_keyframes=1))
        config = structs.FilterConfig(
            n_particles=n_particles, seed=2,
            init=structs.UniformBoxInit(-50000.0, 50000.0, -50000.0, 50000.0),
            position_roughening_m=15.0, heading_roughening_deg=1.0,
            checkpoint_every=1000,
            proposal=structs.ProposalConfig(enabled=use_proposal))
        history = pf.run_filter(config, data.catalog, data.odometry,
                                data.measurements, data.tables)
        errors = metrics.map_position_errors_m(history.health, data.truth)
        converged = errors < radius_m
        for keyframe_idx in range(len(errors)):
            if bool(np.all(converged[keyframe_idx:])):
                return keyframe_idx
        return len(errors)

    def test_converges_where_brute_force_cannot(self):
        """A 100x100 km box at 1500 particles is ~1 particle per 6.7 km^2.
        Sampling cannot cover that; resection does not need to.

        Note how weak the contrast is at easier settings: at 4000 particles
        over a 50 km box brute force also converges (by keyframe ~4), because
        uniform heading means a percent or so of particles happen to match
        any given bearing and the arc structure does the rest. The claim
        being tested is about *scaling*, so the test has to be run where
        area actually beats particle count."""
        with_proposal = self._time_to_converge(1500, use_proposal=True)
        brute_force = self._time_to_converge(1500, use_proposal=False)
        self.assertLess(with_proposal, 15,
                        f"proposal-based init took {with_proposal} keyframes")
        self.assertGreater(brute_force, 100,
                           f"brute force converged in {brute_force} keyframes "
                           f"— the comparison is not measuring anything")


class DoNoHarmTest(unittest.TestCase):
    def _healthy_run(self, enabled):
        data = scenario.generate(
            scenario.harbor_loop(max_visible_range_m=10000.0, keyframe_period_s=_PERIOD_S))
        start = data.truth[0]
        config = structs.FilterConfig(
            n_particles=8000, seed=5,
            init=structs.GaussianInit(start.east_m + 300.0,
                                      start.north_m - 200.0, 500.0),
            checkpoint_every=1000,
            proposal=structs.ProposalConfig(enabled=enabled, on_init=False))
        history = pf.run_filter(config, data.catalog, data.odometry,
                                data.measurements, data.tables)
        return history, data.truth

    def test_does_not_fire_or_degrade_when_healthy(self):
        """A recovery mechanism that fires during normal tracking is worse
        than none: it keeps discarding a belief that was working."""
        enabled, truth = self._healthy_run(enabled=True)
        disabled, _ = self._healthy_run(enabled=False)

        self.assertEqual([e.trigger for e in enabled.proposal_events], [],
                         "proposal fired on a healthy run")
        error_enabled = float(np.median(
            metrics.position_errors_m(enabled.health, truth)[-20:]))
        error_disabled = float(np.median(
            metrics.position_errors_m(disabled.health, truth)[-20:]))
        self.assertAlmostEqual(error_enabled, error_disabled, delta=1e-6)

    def test_clutter_does_not_manufacture_a_confident_fix(self):
        """All-clutter input drives null-share up, which is exactly the
        kidnap trigger. The proposal will fire — it must not then produce a
        confident wrong answer out of meaningless bearings."""
        data = scenario.generate(scenario.harbor_loop(
            max_visible_range_m=10000.0, keyframe_period_s=_PERIOD_S, clutter_only=True))
        start = data.truth[0]
        config = structs.FilterConfig(
            n_particles=8000, seed=5,
            init=structs.GaussianInit(start.east_m, start.north_m, 300.0),
            checkpoint_every=1000)
        history = pf.run_filter(config, data.catalog, data.odometry,
                                data.measurements, data.tables)
        errors = metrics.position_errors_m(history.health, data.truth)
        final_std = history.health[-1].position_std_m
        self.assertFalse(final_std < 100.0 and float(errors[-1]) > 400.0,
                         f"proposal manufactured a confident wrong fix from "
                         f"clutter: std={final_std:.0f} m "
                         f"err={errors[-1]:.0f} m")


class ProvenancePlumbingTest(unittest.TestCase):
    def test_particles_and_events_agree(self):
        data, truth, measurements = _kidnapped_scenario()
        config = structs.FilterConfig(
            n_particles=8000, seed=5,
            init=structs.GaussianInit(data.truth[0].east_m,
                                      data.truth[0].north_m, 300.0),
            checkpoint_every=1000)
        history = pf.run_filter(config, data.catalog, data.odometry,
                                measurements, data.tables)
        fired = [e for e in history.proposal_events if e.n_injected]
        self.assertGreater(len(fired), 0)

        belief = history.final_belief
        event_ids = set(np.unique(belief.proposal_event_id)) - {-1}
        known = {e.event_id for e in fired}
        self.assertTrue(event_ids <= known,
                        f"particles cite unknown events: {event_ids - known}")
        for event in fired:
            self.assertEqual(len(event.hypothesis_landmark_ids),
                             event.n_hypotheses)
            from_event = belief.proposal_hypothesis[
                belief.proposal_event_id == event.event_id]
            if from_event.size:
                self.assertLess(int(from_event.max()), event.n_hypotheses)
                self.assertGreaterEqual(int(from_event.min()), 0)

    def test_health_reports_proposal_share(self):
        data, truth, measurements = _kidnapped_scenario()
        config = structs.FilterConfig(
            n_particles=8000, seed=5,
            init=structs.GaussianInit(data.truth[0].east_m,
                                      data.truth[0].north_m, 300.0),
            checkpoint_every=1000)
        history = pf.run_filter(config, data.catalog, data.odometry,
                                measurements, data.tables)
        shares = [r.proposal_weight_share for r in history.health]
        self.assertTrue(all(0.0 <= s <= 1.0 + 1e-9 for s in shares))
        self.assertGreater(max(shares), 0.1,
                           "no keyframe shows proposal-descended mass")
        fired_keyframes = [r.keyframe_idx for r in history.health
                           if r.proposal_event_id is not None]
        self.assertEqual(fired_keyframes,
                         [e.keyframe_idx for e in history.proposal_events
                          if e.n_injected])


if __name__ == "__main__":
    unittest.main()
