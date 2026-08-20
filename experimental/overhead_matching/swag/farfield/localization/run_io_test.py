import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    run_io,
    structs,
)


class FakeBelief:
    def __init__(self, n=8):
        rng = np.random.default_rng(0)
        self.east_m = rng.normal(0.0, 100.0, n)
        self.north_m = rng.normal(0.0, 100.0, n)
        self.heading_rad = rng.uniform(-np.pi, np.pi, n)
        self.log_weight = np.full(n, -np.log(n))
        self.proposal_event_id = np.full(n, -1)
        self.proposal_hypothesis = np.full(n, -1)
        self.mode_id = np.zeros(n, dtype=int)


class FakeHistory:
    def __init__(self):
        self.health = [structs.HealthRecord(
            keyframe_idx=k, ess=100.0, resampled=False, mean_east_m=0.0,
            mean_north_m=0.0, mean_heading_deg=0.0, map_east_m=0.0,
            map_north_m=0.0, map_heading_deg=0.0, position_std_m=10.0,
            heading_std_deg=5.0, n_measurements=0) for k in range(3)]
        self.proposal_events = [structs.ProposalEvent(
            event_id=0, keyframe_idx=1, trigger="init", n_hypotheses=4,
            n_injected=100, n_tracklets_considered=3,
            n_combinations_examined=10, n_combinations_skipped=0)]
        self.mode_events = [structs.ModeEvent(
            keyframe_idx=1, kind="birth", mode_id=0)]
        self.checkpoints = {0: FakeBelief(), 2: FakeBelief()}


def make_manifest(**overrides):
    fields = dict(
        schema_version=structs.SCHEMA_VERSION,
        scenario_name="tiny",
        anchor_lat_deg=42.35,
        anchor_lon_deg=-71.05,
        n_keyframes=3,
        filter_config=structs.FilterConfig(
            n_particles=8, seed=1,
            init=structs.GaussianInit(0.0, 0.0, 100.0)),
        landmarks=[structs.LandmarkEntry("osm:node:1", 42.36, -71.05, "x")],
        matcher_version="m",
        max_visible_range_m=10000.0,
        export_dir="synthetic:tiny",
        git_commit="deadbeef",
        argv=["run_export", "--flag"],
        created="2026-08-20T00:00:00+00:00",
    )
    fields.update(overrides)
    return structs.RunManifest(**fields)


class RoundTripTest(unittest.TestCase):
    def test_round_trip(self):
        manifest = make_manifest()
        history = FakeHistory()
        truth = [structs.TruthPose(k, 0.0, 0.0, 0.0) for k in range(3)]
        odometry = [structs.OdometryDelta(k, 40.0, 0.0, 0.0, 1.0, 0.02)
                    for k in (1, 2)]
        measurements = [structs.TrackletMeasurement("T1", 1, 45.0, 100.0)]
        tables = {"T1": structs.CompatibilityTable(
            "T1", "m", [], 0.0, -4.0, 4.0, "fast")}

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_io.write_run(run_dir, manifest, truth, odometry,
                             measurements, tables, history)
            loaded = run_io.read_run(run_dir)

        self.assertEqual(loaded.manifest, manifest)
        self.assertEqual(loaded.truth, truth)
        self.assertEqual(loaded.odometry, odometry)
        self.assertEqual(loaded.measurements, measurements)
        self.assertEqual(loaded.tables, tables)
        self.assertEqual(loaded.health, history.health)
        self.assertEqual(loaded.proposal_events, history.proposal_events)
        self.assertEqual(loaded.mode_events, history.mode_events)
        self.assertEqual(set(loaded.checkpoints), {0, 2})
        np.testing.assert_array_equal(
            loaded.checkpoints[0]["east_m"], history.checkpoints[0].east_m)

    def test_manifest_provenance_is_validated_before_writing(self):
        history = FakeHistory()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            for bad in (make_manifest(export_dir=""),
                        make_manifest(git_commit=""),
                        make_manifest(created="")):
                with self.assertRaises(ValueError):
                    run_io.write_run(run_dir, bad, [], [], [], {}, history)
                self.assertFalse(run_dir.exists())  # nothing half-written

    def test_schema_mismatch_on_read(self):
        manifest = make_manifest(schema_version="0.1")
        # write_run does not check schema (a producer always writes its own);
        # read_run refuses foreign versions.
        history = FakeHistory()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_io.write_run(run_dir, manifest, [], [], [], {}, history)
            with self.assertRaises(ValueError):
                run_io.read_run(run_dir)

    def test_health_records_survive_association_payload(self):
        import msgspec
        record = structs.HealthRecord(
            keyframe_idx=3, ess=42.0, resampled=True, mean_east_m=1.0,
            mean_north_m=2.0, mean_heading_deg=3.0, position_std_m=4.0,
            map_east_m=1.0, map_north_m=2.0, map_heading_deg=3.0,
            heading_std_deg=1.5, n_measurements=1,
            associations=[structs.AssociationPosterior(
                tracklet_id="trk_a", anchor_keyframe_idx=3, null_share=0.1,
                responsibilities={"lm_a": 0.9})])
        encoded = msgspec.json.encode(record)
        decoded = msgspec.json.decode(encoded, type=structs.HealthRecord)
        self.assertEqual(decoded, record)


if __name__ == "__main__":
    unittest.main()
