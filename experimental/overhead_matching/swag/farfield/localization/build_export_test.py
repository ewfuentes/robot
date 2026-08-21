import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import shapely

from experimental.overhead_matching.swag.farfield import (
    geometry as geo,
    testing,
)
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.localization import (
    build_export,
    export_ingest,
    structs,
)
from experimental.overhead_matching.swag.farfield.tracking import tracklets

PANO_W = 64  # testing.make_dataset default pano size


def write_tracks(run_dir: Path, name: str, tracks: list) -> None:
    (run_dir / f"tracks_{name}.json").write_text(json.dumps(
        {"range": {"name": name}, "tracks": tracks, "config": {}}))


def write_audits(run_dir: Path, verdict_by_track: dict) -> None:
    audit_dir = run_dir / "semantic_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    meta, lines = {}, []
    for track_id, verdict in verdict_by_track.items():
        key = f"audit_{track_id}"
        meta[key] = {"track_id": track_id}
        payload = {"verdict": verdict, "valid_segments": None}
        lines.append(json.dumps({
            "key": key,
            "response": {"candidates": [{"content": {"parts": [{
                "text": json.dumps(payload)}]}}]}}))
    (audit_dir / "audit_meta.json").write_text(json.dumps(meta))
    (audit_dir / "results.jsonl").write_text("\n".join(lines) + "\n")


def simple_track(track_id: int, pano_x: float, n_keyframes: int = 3) -> dict:
    return {
        "track_id": track_id,
        "birth_keyframe": 0,
        "end_keyframe": n_keyframes - 1,
        "close_reason": "end_of_range",
        "records": [
            {"keyframe": kf, "mask_bbox_window": [pano_x - 4, 10,
                                                  pano_x + 4, 20],
             "window_origin": [0, 0], "action": "propagate"}
            for kf in range(n_keyframes)],
    }


class ResolveMountOffsetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"
        self.run_dir.mkdir()
        self.base = Path(self.tmp.name) / "ds"

    def tearDown(self):
        self.tmp.cleanup()

    def test_override_wins(self):
        offset, source = build_export.resolve_mount_offset(
            self.run_dir, {}, self.base, 123.0)
        self.assertEqual(offset, 123.0)
        self.assertEqual(source, "--mount_offset_deg")

    def test_validated_metadata_outranks_sidecars(self):
        meta = testing.default_metadata()  # 214.0, accuracy_validated
        (self.run_dir / "mount_offset_sweep.json").write_text(json.dumps(
            {"usable": True, "frame": geo.MOUNT_OFFSET_FRAME,
             "mount_offset_deg": 30.0, "verdict": "STRONG",
             "tracklets_used": 9}))
        offset, source = build_export.resolve_mount_offset(
            self.run_dir, meta, self.base, None)
        self.assertEqual(offset, 214.0)
        self.assertIn("accuracy_validated", source)

    def test_sun_sidecar_outranks_sweep(self):
        (self.run_dir / "sun_offset_check.json").write_text(json.dumps(
            {"usable": True, "frame": geo.MOUNT_OFFSET_FRAME,
             "mount_offset_deg": 215.0, "verdict": "AGREEING"}))
        (self.run_dir / "mount_offset_sweep.json").write_text(json.dumps(
            {"usable": True, "frame": geo.MOUNT_OFFSET_FRAME,
             "mount_offset_deg": 30.0, "verdict": "STRONG",
             "tracklets_used": 9}))
        offset, source = build_export.resolve_mount_offset(
            self.run_dir, {}, self.base, None)
        self.assertEqual(offset, 215.0)
        self.assertIn("sun_offset_check", source)

    def test_unusable_or_wrong_frame_sidecars_are_ignored(self):
        # The FIXED-OBJECT abstention and a column-0-frame sidecar both must
        # not reach the export.
        (self.run_dir / "sun_offset_check.json").write_text(json.dumps(
            {"usable": False, "frame": geo.MOUNT_OFFSET_FRAME,
             "mount_offset_deg": 35.0, "verdict": "FIXED-OBJECT"}))
        (self.run_dir / "mount_offset_sweep.json").write_text(json.dumps(
            {"usable": True, "frame": "column_0",
             "mount_offset_deg": 30.0, "verdict": "STRONG",
             "tracklets_used": 9}))
        with self.assertRaises(SystemExit):
            build_export.resolve_mount_offset(self.run_dir, {}, self.base,
                                              None)

    def test_unvalidated_metadata_is_last_resort(self):
        meta = testing.default_metadata()
        meta["mount_offset"]["accuracy_validated"] = False
        offset, source = build_export.resolve_mount_offset(
            self.run_dir, meta, self.base, None)
        self.assertEqual(offset, 214.0)
        self.assertIn("pipeline_metadata", source)


class LoadTracksTest(unittest.TestCase):
    def test_loads_all_ranges_and_refuses_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_tracks(run_dir, "a", [simple_track(1, 30.0)])
            write_tracks(run_dir, "b", [simple_track(2, 40.0)])
            tracks = build_export.load_tracks(run_dir)
            self.assertEqual(set(tracks), {1, 2})
            write_tracks(run_dir, "c", [simple_track(2, 50.0)])
            with self.assertRaises(SystemExit):
                build_export.load_tracks(run_dir)

    def test_missing_tracks_is_a_pointed_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx:
                build_export.load_tracks(Path(tmp))
            self.assertIn("tracking stage", str(ctx.exception))


class BodyFrameTest(unittest.TestCase):
    def test_offset_applied_and_duplicates_refused(self):
        m = tracklets.Measurement("T1", 3, 220.0, 50.0)
        out = build_export.body_frame_measurements([m], 214.0)
        self.assertAlmostEqual(out[0].bearing_body_deg, 6.0)
        self.assertEqual(out[0].kappa, 50.0)
        with self.assertRaises(SystemExit):
            build_export.body_frame_measurements([m, m], 214.0)


class EndToEndTest(unittest.TestCase):
    """Full synthetic pipeline: dataset + tracks + audit + sun sidecar +
    feather -> export directory that export_ingest.load accepts."""

    def test_build_and_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # No metadata offset: the run's own sun sidecar must drive
            # (a validated metadata offset would outrank it by design).
            meta = testing.default_metadata()
            del meta["mount_offset"]
            base = testing.make_dataset(root / "datasets" / "tiny_harbor",
                                        n_frames=4, metadata=meta)
            run_dir = root / "run"
            run_dir.mkdir()
            write_tracks(run_dir, "full",
                         [simple_track(1, PANO_W / 2, 4),
                          simple_track(2, PANO_W * 0.75, 4),
                          simple_track(3, 10.0, 4)])
            write_audits(run_dir, {1: "keep", 2: "keep", 3: "drop"})
            (run_dir / "sun_offset_check.json").write_text(json.dumps(
                {"usable": True, "frame": geo.MOUNT_OFFSET_FRAME,
                 "mount_offset_deg": 0.0, "verdict": "AGREEING"}))

            feather = root / "cat.feather"
            schema.build_frame(
                ids=["('node', 1)"],
                geometries=[shapely.Point(testing.ANCHOR_LON,
                                          testing.ANCHOR_LAT + 0.01)],
                landmark_types=["osm"],
                tags=[{"man_made": "lighthouse"}],
            ).to_feather(feather)

            out = run_dir / "localization_export"
            argv = ["build_export",
                    "--dataset_base", str(base),
                    "--run_dir", str(run_dir),
                    "--output_dir", str(out),
                    "--tables", "uninformative",
                    "--feather", str(feather),
                    "--epoch_keyframes", "5",
                    "--bearing_sigma_deg", "1.0",
                    "--default_log_lr", "0.0",
                    "--clip", "4.0",
                    "--min_step_m", "2.0",
                    "--sigma_pair_m", "1.0",
                    "--max_visible_range_m", "10000"]
            with mock.patch("sys.argv", argv):
                build_export.main()

            data = export_ingest.load(out, max_visible_range_m=10000.0)
            raw = json.loads((out / "export_meta.json").read_text())
        # Track 3 had verdict=drop: only T1 and T2 export.
        self.assertEqual({m.tracklet_id for m in data.measurements},
                         {"T1", "T2"})
        self.assertEqual(data.meta.mount_offset_frame,
                         geo.MOUNT_OFFSET_FRAME)
        self.assertEqual(data.meta.matcher_version,
                         build_export.UNINFORMATIVE_MATCHER)
        self.assertEqual(len(data.tables), 2)
        self.assertEqual(data.n_keyframes, 4)
        # Offset 0 -> body == camera bearing; T1 was pano-centred (az 0).
        t1 = [m for m in data.measurements if m.tracklet_id == "T1"][0]
        self.assertAlmostEqual(t1.bearing_body_deg, 0.0, places=6)
        # Provenance extras land in the raw meta JSON.
        self.assertIn("git_commit", raw)
        self.assertIn("argv", raw)
        self.assertEqual(raw["epoch_keyframes"], 5)
        self.assertEqual(raw["audit_dropped_tracklets"], [3])

    def test_matching_tables_mode_requires_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = testing.make_dataset(root / "datasets" / "tiny_harbor",
                                        n_frames=4)
            run_dir = root / "run"
            run_dir.mkdir()
            write_tracks(run_dir, "full", [simple_track(1, PANO_W / 2, 4)])
            write_audits(run_dir, {1: "keep"})
            feather = root / "cat.feather"
            schema.build_frame(
                ids=["('node', 1)"],
                geometries=[shapely.Point(testing.ANCHOR_LON,
                                          testing.ANCHOR_LAT + 0.01)],
                landmark_types=["osm"],
                tags=[{"man_made": "lighthouse"}],
            ).to_feather(feather)
            # A tables file covering the WRONG tracklet id.
            import msgspec
            from common.python.serialization import msgspec_enc_hook
            tables_path = root / "compatibility.json"
            tables_path.write_bytes(msgspec.json.encode(
                [structs.CompatibilityTable("T99", "llm_v1", [], 0.0, -4.0,
                                            4.0, "fast")],
                enc_hook=msgspec_enc_hook))
            argv = ["build_export",
                    "--dataset_base", str(base),
                    "--run_dir", str(run_dir),
                    "--output_dir", str(root / "out"),
                    "--tables", str(tables_path),
                    "--feather", str(feather),
                    "--mount_offset_deg", "0.0",
                    "--epoch_keyframes", "5",
                    "--bearing_sigma_deg", "1.0",
                    "--default_log_lr", "0.0",
                    "--clip", "4.0",
                    "--min_step_m", "2.0",
                    "--sigma_pair_m", "1.0",
                    "--max_visible_range_m", "10000"]
            with mock.patch("sys.argv", argv), \
                    self.assertRaises(SystemExit) as ctx:
                build_export.main()
            self.assertIn("no table", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
