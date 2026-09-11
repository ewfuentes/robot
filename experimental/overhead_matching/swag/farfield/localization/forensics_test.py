"""Tests for derived event bookmarks and the truth-privileged triage.

The triage assigns blame, so it has to be tested against worlds where blame is
known by construction rather than inferred. Each test builds one specific fault
and asserts that exactly that fault is named:

  T-F-A  bearings no catalog landmark can explain -> geometry-unexplained
  T-F-B  explicable bearings the matcher endorses nothing for -> no-evidence
  T-F-C  explicable bearings whose endorsed set excludes the explanation ->
         matcher-fault
  T-F-D  explicable and endorsed but the filter believed something else ->
         filter-fault
  T-F-E  everything agreeing -> consistent
  T-F-F  a matcher endorsing something geometrically wrong is flagged as
         anti-evidence, not merely as a miss
  T-F-G  a dense catalog is reported as ambiguous rather than silently
         producing a confident verdict — the failure mode that made the first
         implementation of this panel meaningless

Getting these backwards would be worse than having no triage: the panel exists
to point at a subsystem, and pointing at the wrong one costs days.

The fixtures give tracklets several epochs from *moving* truth poses, because
multi-epoch consistency is the only thing that identifies a landmark; a
single-epoch fixture would pass while testing nothing.
"""

import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    filter_catalog as catalog_mod,
    forensics,
    run_io,
    structs,
)


def _catalog(positions):
    """positions: {landmark_id: (east_m, north_m)}."""
    ids = list(positions)
    east = np.array([positions[i][0] for i in ids], dtype=float)
    north = np.array([positions[i][1] for i in ids], dtype=float)
    return catalog_mod.LandmarkCatalog(ids, east, north,
                                       max_visible_range_m=20000.0)


def _table(tracklet_id, entries, default_log_lr=-4.0, status="fast"):
    return structs.CompatibilityTable(
        tracklet_id=tracklet_id, matcher_version="test",
        entries=[structs.CompatibilityEntry(lid, lr) for lid, lr in entries],
        default_log_lr=default_log_lr, clip_lo=-4.0, clip_hi=4.0,
        status=status)


def _manifest(scenario_name, n_keyframes, n_particles=1000):
    """A schema-0.3 manifest: every provenance field explicit."""
    return structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION, dataset="synthetic",
        scenario_name=scenario_name, run_kind="synthetic",
        initialization_kind="test", bearings_consumed=True,
        proposal_enabled=True, localization_inputs_manifest_sha256=None,
        anchor_lat_deg=42.0, anchor_lon_deg=-71.0, n_keyframes=n_keyframes,
        filter_config=structs.FilterConfig(
            n_particles=n_particles, seed=0,
            init=structs.GaussianInit(0.0, 0.0, 100.0)),
        landmarks=[], matcher_version="test",
        max_visible_range_m=20000.0,
        export_dir=f"synthetic:{scenario_name}",
        git_commit="test", argv=["forensics_test"],
        created="2026-08-21T00:00:00+00:00")


def _run_data(catalog, measurements, tables, truth, associations,
              n_keyframes=None, health_extra=None):
    """A minimal RunData: only what forensics reads."""
    n_keyframes = n_keyframes or (max(t.keyframe_idx for t in truth) + 1)
    health = []
    for keyframe_idx in range(n_keyframes):
        rows = associations.get(keyframe_idx, [])
        health.append(structs.HealthRecord(
            keyframe_idx=keyframe_idx, ess=1000.0, resampled=False,
            mean_east_m=0.0, mean_north_m=0.0, mean_heading_deg=0.0,
            map_east_m=0.0, map_north_m=0.0, map_heading_deg=0.0,
            position_std_m=50.0, heading_std_deg=1.0,
            n_measurements=len(rows), associations=rows,
            **(health_extra or {})))
    return run_io.RunData(manifest=_manifest("triage_fixture", n_keyframes),
                          truth=truth, odometry=[],
                          measurements=measurements, tables=tables,
                          health=health, checkpoints={})


def _assoc(tracklet_id, keyframe_idx, responsibilities, null_share=0.1):
    return structs.AssociationPosterior(
        tracklet_id=tracklet_id, anchor_keyframe_idx=keyframe_idx,
        null_share=null_share, responsibilities=responsibilities,
        mode_id=None)


class TableHelpersTest(unittest.TestCase):
    """The one definition of "endorsed" (shared by triage, payload, CLI)."""

    def test_endorsed_entries_clip_before_comparing(self):
        # 10.0 clips to 4.0; the default -1.0 stays; only entries strictly
        # above the clipped default are endorsed.
        table = _table("trk", [("hi", 10.0), ("at_default", -1.0),
                               ("below", -2.0)], default_log_lr=-1.0)
        endorsed = forensics.endorsed_entries(table)
        self.assertEqual(endorsed, {"hi": 4.0})

    def test_table_lookup_returns_clipped_default_for_absent_landmarks(self):
        table = _table("trk", [("hi", 10.0)], default_log_lr=-9.0)
        value, endorsed = forensics.table_lookup(table, "missing")
        self.assertEqual(value, -4.0)  # default clipped to clip_lo
        self.assertFalse(endorsed)
        value, endorsed = forensics.table_lookup(table, "hi")
        self.assertEqual(value, 4.0)
        self.assertTrue(endorsed)

    def test_no_table_means_no_endorsements(self):
        self.assertEqual(forensics.endorsed_entries(None), {})
        self.assertEqual(forensics.table_lookup(None, "x"), (None, False))


class TriageVerdictTest(unittest.TestCase):
    """Fixtures: the vessel runs north up the east=0 line from north=0 to
    north=1200 m, heading north throughout. `true_lm` sits due east of the
    start at (2000, 0); `decoy_lm` sits at (0, 4000), dead ahead.

    Because the vessel moves, the bearing to `true_lm` sweeps from 90 deg to
    121 deg over the leg, and only a landmark near (2000, 0) reproduces that
    whole sweep. That sweep is what makes the tracklet identifiable at all, and
    is why these fixtures are not single-pose.
    """
    N_EPOCHS = 5

    def setUp(self):
        self.catalog = _catalog({"true_lm": (2000.0, 0.0),
                                 "decoy_lm": (0.0, 4000.0)})
        self.truth = [
            structs.TruthPose(keyframe_idx=k, east_m=0.0,
                              north_m=300.0 * k,
                              course_world_cw_deg=0.0)
            for k in range(self.N_EPOCHS)]

    def _bearing_to(self, landmark_id, keyframe_idx):
        """The body bearing a perfect tracker would report."""
        index = self.catalog.index_of(landmark_id)
        pose = self.truth[keyframe_idx]
        east = self.catalog.east_m[index] - pose.east_m
        north = self.catalog.north_m[index] - pose.north_m
        return (math.degrees(math.atan2(east, north))
                - pose.course_world_cw_deg) % 360.0

    def _measurements(self, bearings):
        return [structs.TrackletMeasurement(
            tracklet_id="trk", anchor_keyframe_idx=k,
            bearing_forward_cw_deg=bearings[k], kappa=3000.0)
            for k in range(self.N_EPOCHS)]

    def _triage(self, bearings, entries, responsibilities, catalog=None,
                truth=None):
        catalog = catalog or self.catalog
        truth = truth or self.truth
        data = _run_data(catalog, self._measurements(bearings),
                         {"trk": _table("trk", entries)}, truth,
                         {k: [_assoc("trk", k, responsibilities)]
                          for k in range(self.N_EPOCHS)})
        return forensics.triage_tracklets(data, catalog)["trk"]

    def _true_bearings(self):
        return [self._bearing_to("true_lm", k) for k in range(self.N_EPOCHS)]

    def test_fixture_actually_discriminates(self):
        """Guard on the fixture itself: if a constant bearing also fitted
        `true_lm`, every verdict below would be vacuous."""
        sweep = self._true_bearings()
        self.assertGreater(max(sweep) - min(sweep), 20.0,
                           "the vessel must move enough for the bearing to "
                           "sweep, or nothing is identifiable")

    def test_inconsistent_bearings_are_geometry_unexplained(self):
        """T-F-A. A constant 45 deg bearing while the vessel moves 1.2 km
        cannot be a fixed object: no catalog row explains all five epochs."""
        triage = self._triage([45.0] * self.N_EPOCHS, [("true_lm", 4.0)],
                              {"true_lm": 0.9})
        self.assertEqual(triage.verdict, "geometry-unexplained")
        self.assertFalse(triage.geometry_explicable)
        self.assertGreater(triage.best_fit.rms_deg, triage.tolerance_deg)

    def test_explicable_bearings_with_an_empty_table_are_no_evidence(self):
        """T-F-B. Silence from the matcher is not the same as being wrong,
        and conflating them would blame the matcher for 7 harbour tracklets
        that simply had no table."""
        triage = self._triage(self._true_bearings(), [], {"true_lm": 0.9})
        self.assertEqual(triage.verdict, "no-evidence")
        self.assertEqual(triage.n_endorsed, 0)
        self.assertFalse(triage.anti_evidence)
        self.assertEqual(triage.best_fit.landmark_id, "true_lm")

    def test_endorsing_only_the_wrong_landmark_is_matcher_fault(self):
        """T-F-C, and T-F-F: the endorsed set excludes the explanation, and
        the matcher's best claim is geometrically wrong."""
        triage = self._triage(self._true_bearings(), [("decoy_lm", 4.0)],
                              {"decoy_lm": 0.9})
        self.assertEqual(triage.verdict, "matcher-fault")
        self.assertTrue(triage.anti_evidence)
        self.assertEqual(triage.top_endorsed_fit.landmark_id, "decoy_lm")
        self.assertGreater(triage.top_endorsed_fit.rms_deg,
                           triage.tolerance_deg)
        self.assertEqual(triage.best_fit.landmark_id, "true_lm")

    def test_endorsed_but_unbelieved_is_filter_fault(self):
        """T-F-D. Bearings explicable, matcher endorsed the explanation, and
        the posterior still went elsewhere."""
        triage = self._triage(self._true_bearings(),
                              [("true_lm", 4.0), ("decoy_lm", 3.0)],
                              {"decoy_lm": 0.95})
        self.assertEqual(triage.verdict, "filter-fault")
        self.assertEqual(triage.best_endorsed_fit.landmark_id, "true_lm")
        self.assertLess(triage.best_filter_share,
                        forensics.MEANINGFUL_RESPONSIBILITY)

    def test_all_three_agreeing_is_consistent(self):
        """T-F-E."""
        triage = self._triage(self._true_bearings(), [("true_lm", 4.0)],
                              {"true_lm": 0.92})
        self.assertEqual(triage.verdict, "consistent")
        self.assertLess(triage.best_fit.rms_deg, 0.01)
        self.assertTrue(triage.best_endorsed_fit.endorsed)
        self.assertFalse(triage.anti_evidence)

    def test_a_dense_catalog_is_reported_as_ambiguous(self):
        """T-F-G. The failure that made the first implementation meaningless.
        A ring of landmarks all consistent with a single unmoving observation
        must yield `ambiguous`, not a confident verdict."""
        # Vessel stationary: every landmark along the bearing ray fits.
        stationary = [structs.TruthPose(keyframe_idx=k, east_m=0.0,
                                        north_m=0.0,
                                        course_world_cw_deg=0.0)
                      for k in range(self.N_EPOCHS)]
        ring = {f"lm{i}": (2000.0 + 40.0 * i, 0.0) for i in range(60)}
        catalog = _catalog(ring)
        bearing = 90.0
        triage = self._triage([bearing] * self.N_EPOCHS, [("lm0", 4.0)],
                              {"lm0": 0.9}, catalog=catalog, truth=stationary)
        self.assertTrue(triage.ambiguous)
        self.assertGreater(triage.n_consistent_catalog,
                           forensics.AMBIGUOUS_CATALOG_FITS)
        self.assertIn("ambiguous",
                      forensics.triage_summary({"trk": triage}))

    def test_tolerance_scales_with_declared_precision(self):
        """A tracklet that says sigma 20 deg cannot be held to 5 deg: doing so
        would manufacture unexplained geometry out of admitted imprecision."""
        sharp = self._triage(self._true_bearings(), [("true_lm", 4.0)],
                             {"true_lm": 0.9})
        vague_data = _run_data(
            self.catalog,
            [structs.TrackletMeasurement(
                tracklet_id="trk", anchor_keyframe_idx=k,
                bearing_forward_cw_deg=self._true_bearings()[k],
                kappa=1.0 / math.radians(20.0) ** 2)
             for k in range(self.N_EPOCHS)],
            {"trk": _table("trk", [("true_lm", 4.0)])}, self.truth,
            {k: [_assoc("trk", k, {"true_lm": 0.9})]
             for k in range(self.N_EPOCHS)})
        vague = forensics.triage_tracklets(vague_data, self.catalog)["trk"]
        self.assertLess(sharp.tolerance_deg, vague.tolerance_deg)
        self.assertLessEqual(vague.tolerance_deg,
                             forensics.MAX_RESIDUAL_TOLERANCE_DEG)
        self.assertGreaterEqual(sharp.tolerance_deg,
                                forensics.MIN_RESIDUAL_TOLERANCE_DEG)

    def test_out_of_range_landmarks_cannot_explain_a_tracklet(self):
        """A landmark beyond visibility was not what was seen, so it must not
        be offered as an explanation."""
        catalog = catalog_mod.LandmarkCatalog(
            ["far_lm"], np.array([50000.0]), np.array([0.0]),
            max_visible_range_m=15000.0)
        triage = self._triage([90.0] * self.N_EPOCHS, [("far_lm", 4.0)],
                              {"far_lm": 0.9}, catalog=catalog)
        self.assertEqual(triage.verdict, "geometry-unexplained")
        self.assertIsNone(triage.best_fit)

    def test_candidate_must_be_visible_at_every_epoch(self):
        """One close epoch cannot excuse a candidate absent for the rest."""
        catalog = catalog_mod.LandmarkCatalog(
            ["true_lm"], np.array([2000.0]), np.array([0.0]),
            max_visible_range_m=2100.0)
        triage = self._triage(
            self._true_bearings(), [("true_lm", 4.0)],
            {"true_lm": 0.9}, catalog=catalog)
        self.assertEqual(triage.verdict, "geometry-unexplained")
        self.assertIsNone(triage.best_fit)

    def test_triage_is_empty_without_truth(self):
        data = _run_data(self.catalog, self._measurements([90.0] * 5),
                         {"trk": _table("trk", [("true_lm", 4.0)])},
                         truth=[], associations={}, n_keyframes=5)
        self.assertEqual(forensics.triage_tracklets(data, self.catalog), {})
        self.assertIn("no ground truth", forensics.triage_summary({}))

    def test_triage_is_flagged_truth_privileged(self):
        """The marking must be structural, not a caller's discipline."""
        triage = self._triage(self._true_bearings(), [("true_lm", 4.0)],
                              {"true_lm": 0.92})
        self.assertTrue(triage.truth_privileged)

    def test_summary_separates_no_evidence_from_matcher_fault(self):
        triage = {
            "a": self._triage(self._true_bearings(), [("true_lm", 4.0)],
                              {"true_lm": 0.9}),
            "b": self._triage(self._true_bearings(), [], {"true_lm": 0.9}),
            "c": self._triage(self._true_bearings(), [("decoy_lm", 4.0)],
                              {"decoy_lm": 0.9}),
        }
        summary = forensics.triage_summary(triage)
        self.assertIn("1/3 consistent", summary)
        self.assertIn("1/3 no-evidence", summary)
        self.assertIn("1/3 matcher-fault", summary)
        self.assertIn("1 carry anti-evidence", summary)
        self.assertIn("1 have an empty table", summary)


class DerivedEventTest(unittest.TestCase):
    def _health(self, **overrides):
        base = dict(ess=1000.0, resampled=False, mean_east_m=0.0,
                    mean_north_m=0.0, mean_heading_deg=0.0, map_east_m=0.0,
                    map_north_m=0.0, map_heading_deg=0.0,
                    position_std_m=50.0, heading_std_deg=1.0,
                    n_measurements=0)
        base.update(overrides)
        return base

    def _data(self, health_records, proposal_events=(), mode_events=(),
              n_particles=1000):
        return run_io.RunData(
            manifest=_manifest("events", len(health_records), n_particles),
            truth=[], odometry=[], measurements=[],
            tables={}, health=health_records, checkpoints={},
            proposal_events=list(proposal_events),
            mode_events=list(mode_events))

    def test_map_jump_beyond_reported_sigma_is_flagged(self):
        health = [
            structs.HealthRecord(keyframe_idx=0, **self._health()),
            structs.HealthRecord(keyframe_idx=1,
                                 **self._health(map_east_m=5000.0)),
        ]
        events = forensics.derive_events(self._data(health))
        jumps = [e for e in events if e.kind == "map_jump"]
        self.assertEqual(len(jumps), 1)
        self.assertEqual(jumps[0].keyframe_idx, 1)
        self.assertEqual(jumps[0].source, "derived")
        self.assertIn("5.0 km", jumps[0].label)

    def test_ordinary_motion_under_the_floor_is_not_flagged(self):
        health = [
            structs.HealthRecord(keyframe_idx=0,
                                 **self._health(position_std_m=5.0)),
            structs.HealthRecord(keyframe_idx=1,
                                 **self._health(position_std_m=5.0,
                                                map_east_m=40.0)),
        ]
        events = forensics.derive_events(self._data(health))
        self.assertFalse([e for e in events if e.kind == "map_jump"],
                         "a 40 m step at 5 m sigma is 8 sigma but well under "
                         "the floor, and a confidently-tracking filter must "
                         "not flag every keyframe")

    def test_ess_crash_reports_the_leading_edge_only(self):
        health = [structs.HealthRecord(keyframe_idx=k,
                                       **self._health(ess=10.0 if 2 <= k <= 6
                                                      else 900.0))
                  for k in range(10)]
        crashes = [e for e in forensics.derive_events(self._data(health))
                   if e.kind == "ess_crash"]
        self.assertEqual([e.keyframe_idx for e in crashes], [2])

    def test_resample_storm_needs_sustained_resampling(self):
        health = [structs.HealthRecord(keyframe_idx=k,
                                       **self._health(resampled=True))
                  for k in range(12)]
        storms = [e for e in forensics.derive_events(self._data(health))
                  if e.kind == "resample_storm"]
        self.assertEqual(len(storms), 1)

        occasional = [structs.HealthRecord(
            keyframe_idx=k, **self._health(resampled=(k % 4 == 0)))
            for k in range(12)]
        self.assertFalse([e for e in forensics.derive_events(
            self._data(occasional)) if e.kind == "resample_storm"])

    def test_association_flip_is_detected_and_scoped(self):
        health = [
            structs.HealthRecord(keyframe_idx=0, **self._health(
                associations=[_assoc("trk", 0, {"lm_a": 0.9})])),
            structs.HealthRecord(keyframe_idx=1, **self._health(
                associations=[_assoc("trk", 1, {"lm_b": 0.8})])),
        ]
        flips = [e for e in forensics.derive_events(self._data(health))
                 if e.kind == "association_flip"]
        self.assertEqual(len(flips), 1)
        self.assertIn("lm_a -> lm_b", flips[0].detail)

    def test_negligible_responsibility_is_not_a_flip(self):
        health = [
            structs.HealthRecord(keyframe_idx=0, **self._health(
                associations=[_assoc("trk", 0, {"lm_a": 0.9})])),
            structs.HealthRecord(keyframe_idx=1, **self._health(
                associations=[_assoc("trk", 1, {"lm_b": 0.01})])),
        ]
        self.assertFalse([e for e in forensics.derive_events(self._data(health))
                          if e.kind == "association_flip"])

    def test_gate_rejected_proposal_is_logged_as_a_warning(self):
        health = [structs.HealthRecord(keyframe_idx=0, **self._health())]
        event = structs.ProposalEvent(
            event_id=0, keyframe_idx=0, trigger="null_share", n_hypotheses=9,
            n_injected=0, n_tracklets_considered=3,
            n_combinations_examined=10, n_combinations_skipped=0,
            gate_passed=False, gate_best_hypothesis_nats=1.85,
            gate_reference_nats=-0.92)
        events = forensics.derive_events(
            self._data(health, proposal_events=[event]))
        rejected = [e for e in events if e.kind == "proposal"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].severity, "warn")
        self.assertEqual(rejected[0].source, "logged")
        self.assertIn("1.85", rejected[0].detail)

    def test_events_are_sorted_by_keyframe(self):
        health = [structs.HealthRecord(keyframe_idx=k,
                                       **self._health(ess=10.0))
                  for k in range(5)]
        events = forensics.derive_events(self._data(health))
        self.assertEqual([e.keyframe_idx for e in events],
                         sorted(e.keyframe_idx for e in events))


if __name__ == "__main__":
    unittest.main()
