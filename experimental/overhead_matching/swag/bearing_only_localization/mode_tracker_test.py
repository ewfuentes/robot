"""Mode clustering and genealogy (design doc §5.1/§5.6).

Clustering is the easy half. The half that goes wrong quietly is identity:
a tracker that renumbers modes whenever they move, or hands a surviving
mode's id to the wrong child of a split, produces a genealogy that looks
plausible and means nothing — and the visualizer built on top of it will
confidently mislabel where a mode came from.
"""

import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    filter as pf,
    mode_tracker,
    proposal,
    scenario,
    structs,
)


def _belief(east, north, heading=None, log_weight=None, mode_id=None,
            proposal_event_id=None, proposal_hypothesis=None):
    east = np.asarray(east, dtype=float)
    n = east.shape[0]
    return pf.ParticleBelief(
        east_m=east, north_m=np.asarray(north, dtype=float),
        heading_rad=(np.zeros(n) if heading is None
                     else np.asarray(heading, dtype=float)),
        log_weight=(np.zeros(n) if log_weight is None
                    else np.asarray(log_weight, dtype=float)),
        proposal_event_id=proposal_event_id,
        proposal_hypothesis=proposal_hypothesis,
        mode_id=(np.full(n, -1, dtype=np.int64) if mode_id is None
                 else np.asarray(mode_id, dtype=np.int64)))


def _two_clusters(n=400, separation=3000.0):
    east = np.concatenate([np.full(n, -separation / 2), np.full(n, separation / 2)])
    east = east + np.linspace(-20.0, 20.0, 2 * n)
    return _belief(east, np.zeros(2 * n))


class ClusteringTest(unittest.TestCase):
    def setUp(self):
        self.config = structs.ModeConfig()

    def test_separated_clouds_are_separate_modes(self):
        labels, n_clusters = mode_tracker.cluster(_two_clusters(), self.config)
        self.assertEqual(n_clusters, 2)
        self.assertEqual(len(set(labels[:400])), 1)
        self.assertEqual(len(set(labels[400:])), 1)
        self.assertNotEqual(labels[0], labels[-1])

    def test_one_cloud_is_one_mode(self):
        rng = np.random.default_rng(0)
        belief = _belief(rng.normal(0.0, 40.0, 800),
                         rng.normal(0.0, 40.0, 800))
        labels, n_clusters = mode_tracker.cluster(belief, self.config)
        self.assertEqual(n_clusters, 1)

    def test_heading_separates_modes_at_the_same_place(self):
        """Two hypotheses can share a position and differ only in heading —
        a real ambiguity for a bearing-only filter, and one that a
        position-only clusterer would merge away."""
        n = 400
        belief = _belief(np.zeros(2 * n), np.zeros(2 * n),
                         heading=np.concatenate([np.zeros(n),
                                                 np.full(n, math.pi)]))
        _, n_clusters = mode_tracker.cluster(belief, self.config)
        self.assertEqual(n_clusters, 2)

    def test_heading_axis_wraps(self):
        """Headings either side of north are neighbours, not opposites."""
        n = 400
        heading = np.concatenate([np.full(n, math.radians(1.0)),
                                  np.full(n, math.radians(359.0))])
        belief = _belief(np.zeros(2 * n), np.zeros(2 * n), heading=heading)
        _, n_clusters = mode_tracker.cluster(belief, self.config)
        self.assertEqual(n_clusters, 1)

    def test_is_independent_of_particle_order(self):
        """Determinism: a clusterer whose labels depend on particle order
        would put nondeterminism inside the filter's output (§3.8)."""
        belief = _two_clusters()
        forward, _ = mode_tracker.cluster(belief, self.config)
        order = np.random.default_rng(1).permutation(belief.n)
        shuffled = _belief(belief.east_m[order], belief.north_m[order])
        backward, _ = mode_tracker.cluster(shuffled, self.config)
        # Same partition, up to which label each cluster got.
        self.assertEqual(
            {frozenset(np.nonzero(forward == label)[0])
             for label in np.unique(forward)},
            {frozenset(order[np.nonzero(backward == label)[0]])
             for label in np.unique(backward)})


class GenealogyTest(unittest.TestCase):
    def setUp(self):
        self.config = structs.ModeConfig()
        self.tracker = mode_tracker.ModeTracker(self.config)

    def test_ids_persist_while_a_mode_moves(self):
        belief = _belief(np.linspace(-20.0, 20.0, 400), np.zeros(400))
        first = self.tracker.update(belief, 0)
        self.assertEqual(len(first.modes), 1)
        mode_id = first.modes[0].mode_id

        belief.mode_id = first.mode_id
        belief.east_m = belief.east_m + 900.0  # far further than a cell
        second = self.tracker.update(belief, 1)
        self.assertEqual([m.mode_id for m in second.modes], [mode_id])
        self.assertEqual(second.events, [])
        self.assertEqual(second.modes[0].birth_keyframe_idx, 0)

    def test_split_keeps_the_id_on_the_dominant_child(self):
        belief = _belief(np.linspace(-20.0, 20.0, 400), np.zeros(400))
        first = self.tracker.update(belief, 0)
        original = first.modes[0].mode_id

        # 300 particles stay, 100 leave: the id follows the mass.
        belief.mode_id = first.mode_id
        belief.east_m = np.concatenate([np.linspace(-20.0, 20.0, 300),
                                        np.linspace(4980.0, 5020.0, 100)])
        second = self.tracker.update(belief, 1)
        by_weight = sorted(second.modes, key=lambda m: -m.weight)
        self.assertEqual(by_weight[0].mode_id, original)
        self.assertNotEqual(by_weight[1].mode_id, original)
        births = [e for e in second.events if e.kind == "birth"]
        self.assertEqual([e.mode_id for e in births], [by_weight[1].mode_id])
        self.assertEqual(births[0].parent_mode_ids, [original])

    def test_merge_is_recorded_and_keeps_the_heavier_id(self):
        belief = _two_clusters()
        first = self.tracker.update(belief, 0)
        self.assertEqual(len(first.modes), 2)
        # Make one mode heavier so the surviving id is predictable.
        weights = np.concatenate([np.full(400, math.log(3.0)), np.zeros(400)])
        belief = _two_clusters()
        belief.log_weight = weights
        belief.mode_id = first.mode_id
        heavier = max(first.modes, key=lambda m: (
            belief.normalized_weights()[first.mode_id == m.mode_id].sum()))

        belief.east_m = np.linspace(-20.0, 20.0, 800)  # collapse together
        second = self.tracker.update(belief, 1)
        self.assertEqual(len(second.modes), 1)
        self.assertEqual(second.modes[0].mode_id, heavier.mode_id)
        merges = [e for e in second.events if e.kind == "merge"]
        self.assertEqual(len(merges), 1)
        self.assertEqual(len(merges[0].parent_mode_ids), 2)

    def test_death_is_recorded(self):
        belief = _two_clusters()
        first = self.tracker.update(belief, 0)
        surviving, dying = first.modes[0].mode_id, first.modes[1].mode_id

        kept = first.mode_id == surviving
        belief = _belief(belief.east_m[kept], belief.north_m[kept],
                         mode_id=first.mode_id[kept])
        second = self.tracker.update(belief, 1)
        deaths = [e for e in second.events if e.kind == "death"]
        self.assertEqual([e.mode_id for e in deaths], [dying])

    def test_new_ids_are_never_reused(self):
        belief = _two_clusters()
        seen = set()
        for keyframe in range(4):
            assignment = self.tracker.update(belief, keyframe)
            for mode in assignment.modes:
                seen.add(mode.mode_id)
            # Everything dies each keyframe; ids must not be recycled.
            belief = _two_clusters()
            belief.east_m = belief.east_m + 20000.0 * (keyframe + 1)
        self.assertEqual(len(seen), 8)


class ProvenanceTest(unittest.TestCase):
    def test_birth_from_a_proposal_records_its_hypothesis(self):
        """`[CONTRACT]` §5.5: a mode founded by injected particles must name
        the event, tracklets and landmarks that produced it — that is the
        one-click answer to 'where did this wrong mode come from'."""
        n = 400
        belief = _belief(
            np.linspace(-20.0, 20.0, n), np.zeros(n),
            proposal_event_id=np.full(n, 7, dtype=np.int64),
            proposal_hypothesis=np.full(n, 2, dtype=np.int64))
        event = structs.ProposalEvent(
            event_id=7, keyframe_idx=42, trigger="null_share",
            n_hypotheses=3, n_injected=n, n_tracklets_considered=3,
            n_combinations_examined=9, n_combinations_skipped=0,
            hypothesis_tracklet_ids=[["a"], ["b"], ["trk_x", "trk_y"]],
            hypothesis_landmark_ids=[["p"], ["q"], ["graves", "boston"]])
        tracker = mode_tracker.ModeTracker(structs.ModeConfig())
        assignment = tracker.update(belief, 42, [event])

        provenance = assignment.modes[0].provenance
        self.assertEqual(provenance["source"], "proposal")
        self.assertEqual(provenance["proposal_event_id"], 7)
        self.assertEqual(provenance["hypothesis_index"], 2)
        self.assertEqual(provenance["trigger"], "null_share")
        self.assertEqual(provenance["landmark_ids"], "graves,boston")
        self.assertEqual(provenance["tracklet_ids"], "trk_x,trk_y")

    def test_motion_descended_modes_say_so(self):
        belief = _belief(np.linspace(-20.0, 20.0, 400), np.zeros(400))
        tracker = mode_tracker.ModeTracker(structs.ModeConfig())
        assignment = tracker.update(belief, 0)
        self.assertEqual(assignment.modes[0].provenance,
                         {"source": "motion"})


class FilterIntegrationTest(unittest.TestCase):
    """T-F3 with the modes named rather than inferred from a
    mass-on-each-side count — which a single cloud straddling the axis
    passes trivially.

    The required behaviour: hold every hypothesis the evidence cannot
    separate, and commit exactly when it can. The two-lighthouse world has
    an exact C2 symmetry — rotating everything 180 degrees about the
    landmark midpoint maps the two identical lighthouses onto each other —
    so which half of the behaviour applies depends on whether the MATCHER
    can tell the lighthouses apart. (The archived world-frame motion model
    collapsed even the symmetric-matcher case by kf 60: world-frame deltas
    leak absolute direction of travel, which a GPS-denied system does not
    have. §5.2's honest model must NOT reproduce that.)"""

    @staticmethod
    def _run(tables):
        cfg = scenario.symmetric_pair(keyframe_period_s=5.0)
        data = scenario.generate(cfg)
        config = structs.FilterConfig(
            n_particles=40000, seed=1,
            init=structs.UniformBoxInit(-1500.0, 1500.0, -1200.0, 1200.0),
            checkpoint_every=1000)
        history = pf.run_filter(
            config, data.catalog, data.odometry, data.measurements,
            tables if tables is not None else data.tables)
        return data, history

    def test_symmetric_matcher_holds_both_modes_forever(self):
        """With a matcher that cannot separate the twins, the rotated
        hypothesis is EXACTLY as likely as truth at every keyframe, so the
        honest posterior holds two balanced modes to the end — collapse
        would be manufactured information."""
        cfg = scenario.symmetric_pair(keyframe_period_s=5.0)
        data = scenario.generate(cfg)
        both = [structs.CompatibilityEntry(lm_id, cfg.identity_clip)
                for lm_id in data.landmark_ids]
        tables = {tid: structs.CompatibilityTable(
            tracklet_id=table.tracklet_id,
            matcher_version=table.matcher_version, entries=both,
            default_log_lr=table.default_log_lr, clip_lo=table.clip_lo,
            clip_hi=table.clip_hi, status=table.status)
            for tid, table in data.tables.items()}
        _, history = self._run(tables)

        early = history.health[5]
        self.assertGreaterEqual(len(early.modes), 2,
                                "collapsed to one mode immediately")
        second_weight = sorted(early.modes, key=lambda m: -m.weight)[1].weight
        self.assertGreater(second_weight, 0.15,
                           "the rival hypothesis was never really held")
        # Multimodality shows up as entropy, not as a hidden average.
        self.assertGreater(early.mode_entropy_nats, 0.4)

        final = history.health[-1]
        self.assertEqual(len(final.modes), 2,
                         "the exactly-symmetric rival was killed without "
                         "evidence — absolute direction is leaking in")
        self.assertGreater(final.mode_entropy_nats, 0.6,
                           "modes are no longer balanced")
        truth = data.truth[-1]
        nearest = min(
            math.hypot(m.mean_east_m - truth.east_m,
                       m.mean_north_m - truth.north_m)
            for m in final.modes)
        self.assertLess(nearest, 400.0,
                        "truth is not among the held hypotheses")

    def test_identity_matcher_collapses_to_the_true_mode(self):
        """With identity tables the rotated hypothesis explains every
        bearing with the WRONG lighthouse at default_log_lr — evidence
        exists, and the filter must commit to the right mode."""
        data, history = self._run(None)
        final = history.health[-1]
        self.assertEqual(len(final.modes), 1,
                         "evidence never resolved the ambiguity")
        self.assertLess(final.mode_entropy_nats, 0.05)
        truth = data.truth[-1]
        self.assertLess(math.hypot(final.modes[0].mean_east_m - truth.east_m,
                                   final.modes[0].mean_north_m - truth.north_m),
                        300.0, "collapsed onto the wrong mode")

    def test_per_mode_associations_are_emitted_per_mode(self):
        cfg = scenario.symmetric_pair(keyframe_period_s=5.0)
        data = scenario.generate(cfg)
        both = [structs.CompatibilityEntry(lm_id, cfg.identity_clip)
                for lm_id in data.landmark_ids]
        tables = {tid: structs.CompatibilityTable(
            tracklet_id=table.tracklet_id, matcher_version="v", entries=both,
            default_log_lr=table.default_log_lr, clip_lo=table.clip_lo,
            clip_hi=table.clip_hi, status=table.status)
            for tid, table in data.tables.items()}
        config = structs.FilterConfig(
            n_particles=40000, seed=1,
            init=structs.UniformBoxInit(-1500.0, 1500.0, -1200.0, 1200.0),
            checkpoint_every=1000)
        history = pf.run_filter(config, data.catalog, data.odometry,
                                data.measurements, tables)

        # Find a keyframe with measurements and at least two modes, and check
        # the modes disagree about which landmark the tracklet is.
        disagreements = 0
        for record in history.health:
            per_mode = [a for a in record.associations if a.mode_id is not None]
            if len(per_mode) < 2:
                continue
            for tracklet_id in {a.tracklet_id for a in per_mode}:
                claims = {
                    a.mode_id: max(a.responsibilities,
                                   key=a.responsibilities.get)
                    for a in per_mode if a.tracklet_id == tracklet_id
                    and a.responsibilities}
                if len(set(claims.values())) > 1:
                    disagreements += 1
        self.assertGreater(disagreements, 0,
                           "no keyframe where two modes disagree about a "
                           "tracklet — the per-mode split is not doing work")

    def test_modes_are_deterministic(self):
        data = scenario.generate(scenario.harbor_loop(keyframe_period_s=5.0))
        config = structs.FilterConfig(
            n_particles=4000, seed=3,
            init=structs.GaussianInit(data.truth[0].east_m,
                                      data.truth[0].north_m, 400.0),
            checkpoint_every=1000)
        runs = [pf.run_filter(config, data.catalog, data.odometry,
                              data.measurements, data.tables)
                for _ in range(2)]
        self.assertEqual(
            [(m.mode_id, round(m.weight, 12))
             for m in runs[0].health[-1].modes],
            [(m.mode_id, round(m.weight, 12))
             for m in runs[1].health[-1].modes])
        self.assertEqual(runs[0].mode_events, runs[1].mode_events)


if __name__ == "__main__":
    unittest.main()
