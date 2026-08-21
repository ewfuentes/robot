"""Tests for the Tier-3 replay service and the §7.2 attribution primitive.

The load-bearing claims under test:

  T-R1  Instrumentation is inert. An observed run and an unobserved run
        produce identical histories, so attaching the recorder cannot change
        the thing it measures.
  T-R2  A run directory is a sufficient replay unit: replaying from it alone
        reproduces the recorded particle-history hash.
  T-R3  An under-specified manifest is detected, not silently replayed with
        today's defaults substituted (the failure that made an earlier real
        run unreplayable).
  T-R4  The attribution closes: the per-tracklet decomposition of a group's
        log-odds change agrees with the change measured independently from
        Tier 0.
  T-R5  Edits do what they say, and only that.
  T-R6  A stale attribution cache is rejected rather than trusted.

Replay reads `max_visible_range_m` from the manifest — required since schema
0.3 — so the old "fallback radius" tests became tests that the RECORDED value
is used and that a manifest without one is unreplayable, never guessed at.
"""

import tempfile
import unittest
from pathlib import Path

import msgspec
import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    attribution,
    filter as pf,
    replay as replay_mod,
    run_io,
    scenario,
    structs,
)

VISIBLE_RANGE_M = 10000.0


def _write_run(run_dir: Path, config: structs.FilterConfig, data,
               tables=None):
    tables = data.tables if tables is None else tables
    history = pf.run_filter(config, data.catalog, data.odometry,
                            data.measurements, tables)
    manifest = structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION,
        scenario_name=data.config.name,
        anchor_lat_deg=data.config.anchor_lat_deg,
        anchor_lon_deg=data.config.anchor_lon_deg,
        n_keyframes=data.n_keyframes,
        filter_config=config,
        landmarks=data.config.landmarks,
        matcher_version=scenario.MATCHER_VERSION,
        particle_history_sha256=history.particle_history_sha256,
        max_visible_range_m=data.config.max_visible_range_m,
        export_dir=f"synthetic:{data.config.name}",
        git_commit="test", argv=["replay_test"],
        created="2026-08-21T00:00:00+00:00")
    run_io.write_run(run_dir, manifest, data.truth, data.odometry,
                     data.measurements, tables, history)
    return history


def _harbor_run(tmp: Path, **config_kwargs):
    """A small multimodal-capable run: enough modes to attribute between."""
    cfg = scenario.harbor_loop(keyframe_period_s=5.0,
                               max_visible_range_m=VISIBLE_RANGE_M)
    data = scenario.generate(cfg)
    start = data.truth[0]
    config = structs.FilterConfig(
        n_particles=config_kwargs.pop("n_particles", 4000),
        seed=config_kwargs.pop("seed", 3),
        init=structs.GaussianInit(start.east_m, start.north_m, 400.0),
        checkpoint_every=10, **config_kwargs)
    run_dir = tmp / "run"
    history = _write_run(run_dir, config, data)
    return run_dir, history, data, config


class ObserverInertnessTest(unittest.TestCase):
    def test_observer_does_not_change_the_run(self):
        """T-R1. If instrumentation perturbed the filter, every Tier-3 number
        would describe a run that only exists while being watched."""
        cfg = scenario.harbor_loop(keyframe_period_s=5.0,
                                   max_visible_range_m=VISIBLE_RANGE_M)
        data = scenario.generate(cfg)
        start = data.truth[0]
        config = structs.FilterConfig(
            n_particles=2000, seed=7,
            init=structs.GaussianInit(start.east_m, start.north_m, 300.0),
            checkpoint_every=10)

        bare = pf.run_filter(config, data.catalog, data.odometry,
                             data.measurements, data.tables)
        recorder = attribution.AttributionRecorder()
        watched = pf.run_filter(config, data.catalog, data.odometry,
                                data.measurements, data.tables,
                                observer=recorder)

        self.assertEqual(bare.particle_history_sha256,
                         watched.particle_history_sha256)
        self.assertEqual(bare.health, watched.health)
        self.assertEqual(bare.proposal_events, watched.proposal_events)
        self.assertTrue(recorder.contributions,
                        "recorder saw nothing, so the test proves nothing")

    def test_base_observer_hooks_are_all_optional(self):
        """A subclass overriding one hook must not have to define the rest."""
        class OnlyOne(pf.RunObserver):
            def __init__(self):
                self.seen = 0

            def keyframe_end(self, keyframe_idx, belief, health):
                self.seen += 1

        cfg = scenario.straight_leg(speed_mps=20.0, keyframe_period_s=5.0,
                                    epoch_length_keyframes=3,
                                    max_visible_range_m=VISIBLE_RANGE_M)
        data = scenario.generate(cfg)
        config = structs.FilterConfig(
            n_particles=200, seed=1,
            init=structs.GaussianInit(data.truth[0].east_m,
                                      data.truth[0].north_m, 300.0))
        observer = OnlyOne()
        pf.run_filter(config, data.catalog, data.odometry, data.measurements,
                      data.tables, observer=observer)
        self.assertEqual(observer.seen, data.n_keyframes)


class ReplayFidelityTest(unittest.TestCase):
    def test_run_directory_is_a_sufficient_replay_unit(self):
        """T-R2. Tier 1 + manifest, with no access to the original export."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, history, _, _ = _harbor_run(Path(tmp))
            result = replay_mod.replay(run_dir)
            self.assertTrue(result.hash_match)
            self.assertTrue(result.faithful)
            self.assertEqual(result.history.health, history.health)

    def test_recorded_visible_range_is_used(self):
        """The radius feeds proposal geometry, so replay reads the RECORDED
        value — there is no parameter to override it and no fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, _, _ = _harbor_run(Path(tmp))
            inputs = replay_mod.load_inputs(run_dir)
            np.testing.assert_allclose(inputs.catalog.max_visible_range_m,
                                       VISIBLE_RANGE_M)
            self.assertEqual(inputs.max_visible_range_m, VISIBLE_RANGE_M)

    def test_missing_visible_range_is_unreplayable_not_assumed(self):
        """A manifest without the radius cannot exist under schema 0.3; a
        hand-mutilated one must be reported unreplayable, never replayed
        under an assumed radius (the old fallback constant even disagreed
        with the catalog default it claimed to match)."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, _, _ = _harbor_run(Path(tmp))
            raw = msgspec.json.decode(
                (run_dir / "manifest.json").read_bytes())
            del raw["max_visible_range_m"]
            (run_dir / "manifest.json").write_bytes(msgspec.json.encode(raw))

            status = replay_mod.replayability(run_dir)
            self.assertFalse(status.has_max_visible_range)
            self.assertFalse(status.replayable)
            note = " ".join(status.notes)
            self.assertIn("max_visible_range_m", note)
            self.assertNotIn("assume", note)
            # And the strict reader refuses the manifest outright.
            with self.assertRaises(msgspec.ValidationError):
                replay_mod.load_inputs(run_dir)

    def test_under_specified_manifest_is_detected(self):
        """T-R3. A config field added after a run was written must not be
        silently backfilled with today's default — that replays different
        filter semantics under the original run's name."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, _, _ = _harbor_run(Path(tmp))
            raw = msgspec.json.decode(
                (run_dir / "manifest.json").read_bytes())
            del raw["filter_config"]["matcher_recall"]
            del raw["filter_config"]["proposal"]["evidence_gate"]
            (run_dir / "manifest.json").write_bytes(msgspec.json.encode(raw))

            status = replay_mod.replayability(run_dir)
            self.assertFalse(status.replayable)
            self.assertIn("matcher_recall", status.missing_config_keys)
            self.assertIn("proposal.evidence_gate",
                          status.missing_config_keys)
            with self.assertRaises(replay_mod.ReplayDivergence):
                replay_mod.replay(run_dir)

    def test_divergence_is_inspectable_without_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, _, _ = _harbor_run(Path(tmp))
            raw = msgspec.json.decode(
                (run_dir / "manifest.json").read_bytes())
            raw["particle_history_sha256"] = "0" * 64
            (run_dir / "manifest.json").write_bytes(msgspec.json.encode(raw))

            with self.assertRaises(replay_mod.ReplayDivergence):
                replay_mod.replay(run_dir)
            result = replay_mod.replay(run_dir, verify=False)
            self.assertFalse(result.hash_match)
            self.assertIn("DIVERGED", result.report())


class EditsTest(unittest.TestCase):
    def test_dropping_a_tracklet_removes_only_its_measurements(self):
        """T-R5."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, data, _ = _harbor_run(Path(tmp))
            inputs = replay_mod.load_inputs(run_dir)
            victim = data.measurements[0].tracklet_id
            n_victim = sum(1 for m in inputs.measurements
                           if m.tracklet_id == victim)
            self.assertGreater(n_victim, 0)

            edited = replay_mod.apply_edits(
                inputs, replay_mod.Edits(drop_tracklets=(victim,)))
            self.assertEqual(len(edited.measurements),
                             len(inputs.measurements) - n_victim)
            self.assertFalse(any(m.tracklet_id == victim
                                 for m in edited.measurements))
            # Untouched inputs stay untouched.
            self.assertEqual(len(inputs.measurements),
                             n_victim + len(edited.measurements))
            self.assertEqual(edited.config, inputs.config)

    def test_force_landmark_rewrites_the_table_to_one_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, data, _ = _harbor_run(Path(tmp))
            inputs = replay_mod.load_inputs(run_dir)
            tracklet_id = next(iter(inputs.tables))
            target = data.catalog.landmark_ids[0]

            edited = replay_mod.apply_edits(inputs, replay_mod.Edits(
                force_landmark={tracklet_id: target}))
            table = edited.tables[tracklet_id]
            self.assertEqual([e.landmark_id for e in table.entries], [target])
            self.assertEqual(table.entries[0].log_lr, table.clip_hi)
            self.assertIn("forced", table.matcher_version)
            # The original table object is unchanged.
            self.assertNotEqual(inputs.tables[tracklet_id].entries,
                                table.entries)

    def test_log_lr_override_appends_unknown_landmarks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, data, _ = _harbor_run(Path(tmp))
            inputs = replay_mod.load_inputs(run_dir)
            tracklet_id = next(iter(inputs.tables))
            existing = {e.landmark_id
                        for e in inputs.tables[tracklet_id].entries}
            fresh = next(lid for lid in data.catalog.landmark_ids
                         if lid not in existing)

            edited = replay_mod.apply_edits(inputs, replay_mod.Edits(
                log_lr={tracklet_id: {fresh: 2.5}}))
            entries = {e.landmark_id: e.log_lr
                       for e in edited.tables[tracklet_id].entries}
            self.assertEqual(entries[fresh], 2.5)
            self.assertTrue(existing <= set(entries))

    def test_config_edits_are_applied_and_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, _, config = _harbor_run(Path(tmp))
            inputs = replay_mod.load_inputs(run_dir)
            edited = replay_mod.apply_edits(inputs, replay_mod.Edits(
                pi0=0.45, disable_proposal=True, seed=99))
            self.assertEqual(edited.config.pi0, 0.45)
            self.assertEqual(edited.config.seed, 99)
            self.assertFalse(edited.config.proposal.enabled)
            self.assertEqual(edited.config.n_particles, config.n_particles)
            self.assertEqual(inputs.config.pi0, config.pi0)

    def test_empty_edits_round_trips_to_the_same_run(self):
        self.assertTrue(replay_mod.Edits().is_empty)
        self.assertFalse(replay_mod.Edits(pi0=0.3).is_empty)
        self.assertEqual(replay_mod.Edits().describe(), "unmodified")
        self.assertIn("without LT7", replay_mod.Edits(
            drop_tracklets=("LT7",)).describe())

    def test_default_counterfactual_dir_is_inside_the_run(self):
        """REORG rule: nothing writes outside the data root — a ghost lives
        under the run it questions, and the relative name has one owner."""
        edits = replay_mod.Edits(drop_tracklets=("LT7",))
        out = replay_mod.default_counterfactual_dir(Path("/data/run_x"), edits)
        self.assertEqual(out.parent,
                         Path("/data/run_x") / replay_mod.COUNTERFACTUAL_DIRNAME)
        self.assertEqual(out.name, edits.slug())

    def test_counterfactual_writes_a_readable_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run_dir, _, data, _ = _harbor_run(tmp)
            victim = data.measurements[0].tracklet_id
            result = replay_mod.replay(
                run_dir, edits=replay_mod.Edits(drop_tracklets=(victim,)))
            self.assertIsNone(result.hash_match)
            self.assertFalse(result.faithful)

            ghost_dir = tmp / "ghost"
            replay_mod.write_counterfactual(ghost_dir, run_dir, result)
            ghost = run_io.read_run(ghost_dir)
            self.assertIn(victim, ghost.manifest.scenario_name)
            self.assertEqual(ghost.manifest.particle_history_sha256,
                             result.history.particle_history_sha256)
            # The ghost is itself replayable: forensics can recurse.
            self.assertTrue(replay_mod.replay(ghost_dir).hash_match)


class AttributionTest(unittest.TestCase):
    def test_decomposition_closes_against_tier0(self):
        """T-R4. The itemized waterfall must agree with the mode-weight
        trajectory Tier 0 recorded independently. If these disagree, the
        waterfall is a plausible-looking fiction.

        This is the test that caught §7.2's missing term: with only tracklet
        and resample terms, the leading mode's waterfall totalled 0.04 nats
        against a Tier-0 change of 10.7, because mode membership moves when
        the tracker re-clusters."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, _, _ = _harbor_run(Path(tmp))
            cache, result = attribution.compute(run_dir)
            self.assertTrue(result.hash_match)
            self.assertTrue(cache.verified_against_manifest)
            self.assertTrue(cache.contributions)

            # The tight check: the recorder's share and the filter's own mode
            # weight, compared at the same instant, in code that shares
            # nothing but the belief they both describe.
            worst, n_compared = attribution.pointwise_check(cache)
            self.assertGreater(n_compared, 20,
                               "too few comparable points to prove anything")
            self.assertLess(worst, 1e-9,
                            f"recorder and filter disagree about a mode's "
                            f"share by {worst:g}")

            # The looser range-based check, which also exercises `attribute`.
            checked = 0
            for group in {w.group for w in cache.group_weights}:
                shares = sorted(
                    (w.keyframe_idx, w.mass_share)
                    for w in cache.group_weights if w.group == group)
                live = [(kf, s) for kf, s in shares if s > 1e-9]
                if len(live) < 2:
                    continue
                low, high = live[0][0], live[-1][0]
                waterfall = attribution.attribute(cache, group, (low, high))
                if waterfall.observed_nats is None:
                    continue
                checked += 1
                self.assertLess(
                    abs(waterfall.residual_nats), 0.5,
                    f"mode {group} waterfall totals "
                    f"{waterfall.total_nats:.3f} nats but Tier 0 shows "
                    f"{waterfall.observed_nats:.3f} over kf {low}-{high}\n"
                    + waterfall.report())
            self.assertGreater(checked, 0, "no group lived long enough to check")

    def test_structural_and_evidence_terms_are_separated(self):
        """The finding the corrected decomposition exists to expose: how much
        of a mode's rise was evidence and how much was bookkeeping."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, _, _ = _harbor_run(Path(tmp))
            cache, _ = attribution.compute(run_dir)
            kinds = {c.term for c in cache.contributions}
            self.assertIn("tracklet", kinds)
            self.assertIn("recluster", kinds)

            group = max((w.group for w in cache.group_weights))
            waterfall = attribution.attribute(cache, group)
            self.assertAlmostEqual(
                waterfall.evidence_nats + waterfall.structural_nats,
                waterfall.total_nats, places=9)
            bare = attribution.attribute(cache, group,
                                         include_structural=False)
            self.assertAlmostEqual(bare.total_nats, waterfall.evidence_nats,
                                   places=9)

    def test_single_measurement_contribution_matches_hand_computation(self):
        """The primitive itself, checked against the definition rather than
        against another part of the implementation."""
        cfg = scenario.harbor_loop(keyframe_period_s=5.0,
                                   max_visible_range_m=VISIBLE_RANGE_M)
        data = scenario.generate(cfg)
        start = data.truth[0]
        config = structs.FilterConfig(
            n_particles=1500, seed=5,
            init=structs.GaussianInit(start.east_m, start.north_m, 300.0))

        class Spy(pf.RunObserver):
            def __init__(self):
                self.rows = []

            def measurement(self, keyframe_idx, meas, log_weight_before,
                            belief, pass_index):
                from scipy import special
                self.rows.append((
                    keyframe_idx, meas.tracklet_id,
                    float(special.logsumexp(belief.log_weight)
                          - special.logsumexp(log_weight_before))))

        spy = Spy()
        recorder = attribution.AttributionRecorder()

        class Both(pf.RunObserver):
            def measurement(self, *args):
                spy.measurement(*args)
                recorder.measurement(*args)

        pf.run_filter(config, data.catalog, data.odometry, data.measurements,
                      data.tables, observer=Both())

        whole = {(c.keyframe_idx, c.tracklet_id, c.pass_index): c
                 for c in recorder.contributions
                 if c.group == attribution.ALL_GROUPS and c.term == "tracklet"}
        self.assertEqual(len(whole), len(spy.rows))
        for keyframe_idx, tracklet_id, expected in spy.rows:
            row = whole[(keyframe_idx, tracklet_id, 0)]
            self.assertAlmostEqual(row.self_nats, expected, places=9)
            # The whole belief is always 100% of itself, so its share cannot
            # move: relative_nats is 0 for ALL_GROUPS by construction.
            self.assertAlmostEqual(row.relative_nats, 0.0, places=9)

    def test_cache_round_trips_and_rejects_staleness(self):
        """T-R6."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _harbor_run(Path(tmp))
            cache, result = attribution.compute(run_dir)
            attribution.write_cache(run_dir, cache)

            loaded = attribution.read_cache(
                run_dir, expected_sha256=cache.particle_history_sha256)
            self.assertEqual(loaded.contributions, cache.contributions)
            self.assertEqual(loaded.group_weights, cache.group_weights)

            with self.assertRaises(ValueError):
                attribution.read_cache(run_dir, expected_sha256="f" * 64)

    def test_read_cache_is_none_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _harbor_run(Path(tmp))
            self.assertIsNone(attribution.read_cache(run_dir))

    def test_tracklet_series_is_ordered_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _, _, data, _ = _harbor_run(Path(tmp))
            cache, _ = attribution.compute(run_dir)
            tracklet_id = data.measurements[0].tracklet_id
            series = attribution.tracklet_series(cache, tracklet_id)
            self.assertTrue(series)
            keyframes = [row.keyframe_idx for row in series]
            self.assertEqual(keyframes, sorted(keyframes))
            self.assertEqual(len(keyframes), len(set(keyframes)),
                             "a keyframe appears twice: an injection's two "
                             "passes are both being counted")


if __name__ == "__main__":
    unittest.main()
