import math
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.bearing_only_localization import (
    export_ingest,
    gps_to_odometry,
    structs,
)
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    m11_base_export as mbe,
)


def merged_row(tracklet, keyframe, bearing_camera, kappa=100.0, source=0):
    """One row in m6's `merged/measurements.json` shape."""
    return {"tracklet_id": tracklet, "source_track_id": source,
            "anchor_keyframe_idx": keyframe,
            "bearing_camera_deg": bearing_camera, "kappa": kappa}


class BodyFrameMeasurementTest(unittest.TestCase):

    def test_camera_bearing_rotates_by_the_mount_offset(self):
        rows = [merged_row("LT0", 3, 341.69), merged_row("LT1", 5, 10.0)]
        measurements, fused = mbe.body_frame_measurements(rows, 214.0)

        self.assertEqual(fused, 0)
        self.assertAlmostEqual(measurements[0].bearing_body_deg, 127.69,
                               places=6)
        # Wraps rather than going negative: 10 - 214 = -204 -> 156.
        self.assertAlmostEqual(measurements[1].bearing_body_deg, 156.0,
                               places=6)

    def test_kappa_is_carried_through_untouched(self):
        rows = [merged_row("LT0", 3, 100.0, kappa=2777.436346438264)]
        measurements, _ = mbe.body_frame_measurements(rows, 90.0)
        self.assertEqual(measurements[0].kappa, 2777.436346438264)

    def test_two_source_tracks_on_one_keyframe_fuse_into_one_epoch(self):
        # A merged tracklet whose constituents were both alive at keyframe 264
        # writes two rows for the same epoch; export_ingest rejects duplicates,
        # and physically these are one object seen twice.
        rows = [merged_row("LT227_T249", 264, 0.4875, kappa=264.9, source=227),
                merged_row("LT227_T249", 264, 0.5941, kappa=258.3, source=249)]
        measurements, fused = mbe.body_frame_measurements(rows, 0.0)

        self.assertEqual(len(measurements), 1)
        self.assertEqual(fused, 1)
        self.assertTrue(0.4875 < measurements[0].bearing_body_deg < 0.5941)
        # Two agreeing bearings sharpen: the fused concentration is close to
        # the sum, not to either input.
        self.assertGreater(measurements[0].kappa, 500.0)

    def test_opposed_bearings_cancel_instead_of_averaging(self):
        # The failure this guards against is a confident average of two
        # contradictory bearings. The resultant must be weak, not tight.
        rows = [merged_row("LT9", 1, 0.0, kappa=100.0),
                merged_row("LT9", 1, 180.0, kappa=100.0)]
        measurements, _ = mbe.body_frame_measurements(rows, 0.0)
        self.assertLess(measurements[0].kappa, 1e-9)

    def test_non_positive_kappa_is_refused(self):
        for kappa in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                mbe.body_frame_measurements(
                    [merged_row("LT0", 1, 10.0, kappa=kappa)], 0.0)


class TableTest(unittest.TestCase):

    def test_one_flat_table_per_measured_tracklet(self):
        rows = [merged_row("LT0", 1, 10.0), merged_row("LT0", 2, 12.0),
                merged_row("LT1", 2, 90.0)]
        measurements, _ = mbe.body_frame_measurements(rows, 0.0)
        tables = mbe.uninformative_tables(measurements, 0.0, 4.0)

        self.assertEqual([t.tracklet_id for t in tables], ["LT0", "LT1"])
        for table in tables:
            self.assertEqual(table.entries, [])
            self.assertEqual(table.default_log_lr, 0.0)
            self.assertEqual((table.clip_lo, table.clip_hi), (-4.0, 4.0))


class RoundTripTest(unittest.TestCase):
    """The export has to survive the filter's own loader, whose validate() is
    the boundary that would otherwise fail deep inside a run."""

    def build(self, directory: Path, rows: list, n_keyframes: int = 4):
        east = [0.0, 30.0, 60.0, 90.0][:n_keyframes]
        north = [0.0, 0.0, 10.0, 20.0][:n_keyframes]
        measurements, _ = mbe.body_frame_measurements(rows, 214.0)
        mbe.write_export(
            directory,
            meta={"schema_version": structs.SCHEMA_VERSION,
                  "scenario_name": "unit_test",
                  "anchor_lat_deg": 42.3, "anchor_lon_deg": -71.0,
                  "n_keyframes": n_keyframes,
                  "matcher_version": mbe.UNINFORMATIVE_MATCHER,
                  "mount_offset_deg": 214.0,
                  # Provenance keys the reader's struct does not declare; they
                  # must not break decoding.
                  "mount_offset_source": "unit test",
                  "truth_heading_note": "GPS course"},
            landmarks=[structs.LandmarkEntry(landmark_id="osm:node:1",
                                             lat_deg=42.31, lon_deg=-71.0,
                                             type_key="man_made=tower"),
                       structs.LandmarkEntry(landmark_id="osm:node:2",
                                             lat_deg=42.30, lon_deg=-70.99,
                                             type_key="landmark")],
            tables=mbe.uninformative_tables(measurements, 0.0, 4.0),
            measurements=measurements,
            odometry=gps_to_odometry.derive_increments(east, north),
            truth=mbe.truth_poses(east, north, [0.0] * n_keyframes))

    def test_export_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "export"
            self.build(out, [merged_row("LT0", 1, 341.69),
                             merged_row("LT0", 2, 350.0),
                             merged_row("LT1", 3, 12.0)])
            data = export_ingest.load(out)

        self.assertEqual(data.n_keyframes, 4)
        self.assertEqual(len(data.measurements), 3)
        self.assertEqual(len(data.odometry), 3)
        self.assertEqual(len(data.truth), 4)
        self.assertEqual(sorted(data.tables), ["LT0", "LT1"])
        self.assertEqual(data.meta.mount_offset_deg, 214.0)
        self.assertAlmostEqual(data.measurements[0].bearing_body_deg, 127.69,
                               places=6)

    def test_odometry_indices_are_contiguous_from_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "export"
            self.build(out, [merged_row("LT0", 1, 341.69)])
            data = export_ingest.load(out)
        self.assertEqual([o.keyframe_idx for o in data.odometry], [1, 2, 3])

    def test_measurement_beyond_the_last_keyframe_is_rejected(self):
        # A measurement anchored off the end of the run is exactly the kind of
        # misalignment that a silent renumbering would produce.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "export"
            self.build(out, [merged_row("LT0", 9, 341.69)])
            with self.assertRaises(ValueError):
                export_ingest.load(out)


class FuseBearingsTest(unittest.TestCase):

    def test_resultant_is_the_von_mises_product(self):
        bearing, kappa = mbe.fuse_bearings([(10.0, 3.0), (20.0, 4.0)])
        east = 3.0 * math.sin(math.radians(10.0)) + 4.0 * math.sin(
            math.radians(20.0))
        north = 3.0 * math.cos(math.radians(10.0)) + 4.0 * math.cos(
            math.radians(20.0))
        self.assertAlmostEqual(bearing,
                               math.degrees(math.atan2(east, north)) % 360.0)
        self.assertAlmostEqual(kappa, math.hypot(east, north))

    def test_wrap_around_north_is_handled(self):
        # 350 and 10 average to 0, not to 180 -- the trap in naive averaging.
        bearing, _ = mbe.fuse_bearings([(350.0, 5.0), (10.0, 5.0)])
        self.assertAlmostEqual(bearing % 360.0, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()


class ResolveMountOffsetTest(unittest.TestCase):
    """Priority order. The trap this guards: the sweep is relative, so it fits a
    180 deg convention slip perfectly and cannot be used to check itself."""

    def build(self, tmp, *, sweep=None, metadata=None):
        run_dir = Path(tmp) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        if sweep is not None:
            (run_dir / "mount_offset_sweep.json").write_text(
                __import__("json").dumps(sweep))
        meta_path = Path(tmp) / "pipeline_metadata.json"
        meta_path.write_text(__import__("json").dumps(
            {"mount_offset": metadata} if metadata else {}))
        return run_dir, meta_path

    def test_explicit_flag_wins_over_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, meta = self.build(
                tmp, sweep={"usable": True, "mount_offset_deg": 216.0,
                            "verdict": "SMOOTH UNIMODAL", "tracklets_used": 23},
                metadata={"mount_offset_deg": 214.0,
                          "accuracy_validated": True})
            value, source = mbe.resolve_mount_offset(run_dir, meta, 99.0)
        self.assertEqual(value, 99.0)
        self.assertIn("--mount_offset_deg", source)

    def test_validated_metadata_outranks_a_usable_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, meta = self.build(
                tmp, sweep={"usable": True, "mount_offset_deg": 216.0,
                            "verdict": "SMOOTH UNIMODAL", "tracklets_used": 23},
                metadata={"mount_offset_deg": 214.0, "status": "sun_verified",
                          "accuracy_validated": True})
            value, source = mbe.resolve_mount_offset(run_dir, meta, None)
        self.assertEqual(value, 214.0)
        self.assertIn("accuracy_validated", source)

    def test_usable_sweep_beats_unvalidated_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, meta = self.build(
                tmp, sweep={"usable": True, "mount_offset_deg": 4.0,
                            "verdict": "SMOOTH UNIMODAL", "tracklets_used": 17},
                metadata={"mount_offset_deg": 180.0, "status": "operator_prior",
                          "accuracy_validated": False})
            value, source = mbe.resolve_mount_offset(run_dir, meta, None)
        self.assertEqual(value, 4.0)
        self.assertIn("mount_offset_sweep", source)

    def test_unusable_sweep_falls_through_to_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, meta = self.build(
                tmp, sweep={"usable": False, "mount_offset_deg": 277.0,
                            "verdict": "MULTIMODAL", "tracklets_used": 20},
                metadata={"mount_offset_deg": 272.3, "status": "sun_verified",
                          "accuracy_validated": True})
            value, _ = mbe.resolve_mount_offset(run_dir, meta, None)
        self.assertEqual(value, 272.3)

    def test_nothing_available_is_refused_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, meta = self.build(tmp)
            with self.assertRaises(SystemExit):
                mbe.resolve_mount_offset(run_dir, meta, None)


class AuditDroppedTrackletsTest(unittest.TestCase):
    """The export's measurement set must not drift from the matcher's query set:
    both honour `verdict: drop`, and a drift shows up as a measurement with no
    compatibility table two stages later."""

    def build(self, tmp, verdicts, landmarks):
        import json as json_mod
        run_dir = Path(tmp) / "run"
        (run_dir / "semantic_audit").mkdir(parents=True, exist_ok=True)
        (run_dir / "merged").mkdir(parents=True, exist_ok=True)
        meta = {f"T{track}": {"track_id": track}
                for track in {t for lm in landmarks for t in lm["track_ids"]}}
        (run_dir / "semantic_audit" / "audit_meta.json").write_text(
            json_mod.dumps(meta))
        with open(run_dir / "semantic_audit" / "results.jsonl", "w") as handle:
            for key, verdict in verdicts.items():
                handle.write(json_mod.dumps({
                    "key": key,
                    "response": {"candidates": [{"content": {"parts": [
                        {"text": json_mod.dumps({"verdict": verdict})}]}}]},
                }) + "\n")
        (run_dir / "merged" / "landmarks.json").write_text(
            json_mod.dumps(landmarks))
        return run_dir

    def test_drop_verdict_excludes_the_tracklet(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.build(
                tmp, {"T1": "keep", "T2": "drop", "T3": "keep_partial"},
                [{"landmark_id": "LT1", "track_ids": [1]},
                 {"landmark_id": "LT2", "track_ids": [2]},
                 {"landmark_id": "LT3", "track_ids": [3]}])
            self.assertEqual(mbe.audit_dropped_tracklets(run_dir), {"LT2"})

    def test_keep_partial_is_kept(self):
        # m9 queries keep_partial tracklets, so the export must too.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.build(tmp, {"T7": "keep_partial"},
                                 [{"landmark_id": "LT7", "track_ids": [7]}])
            self.assertEqual(mbe.audit_dropped_tracklets(run_dir), set())

    def test_first_audited_constituent_decides(self):
        # Mirrors query_bundles' loop: it breaks on the first track with an
        # audit, so a merged tracklet's verdict is that one's.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.build(tmp, {"T5": "drop", "T6": "keep"},
                                 [{"landmark_id": "LT5_T6",
                                   "track_ids": [5, 6]}])
            self.assertEqual(mbe.audit_dropped_tracklets(run_dir), {"LT5_T6"})

    def test_missing_audit_drops_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                mbe.audit_dropped_tracklets(Path(tmp) / "absent"), set())
