import json
import tempfile
import unittest
from pathlib import Path

import msgspec

from common.python.serialization import msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    run_io,
    structs,
)


def valid_meta(**overrides) -> dict:
    meta = {
        "schema_version": structs.SCHEMA_VERSION,
        "scenario_name": "tiny_harbor",
        "anchor_lat_deg": 42.35,
        "anchor_lon_deg": -71.05,
        "n_keyframes": 3,
        "matcher_version": "uninformative_v1",
        "mount_offset_deg": 214.0,
        "mount_offset_source": "pipeline_metadata (sun_verified)",
        "mount_offset_frame": geo.MOUNT_OFFSET_FRAME,
    }
    meta.update(overrides)
    return meta


def write_export(export_dir: Path, meta=None, measurements=None,
                 tables=None, odometry=None, truth=None,
                 landmarks=None) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "export_meta.json").write_text(
        json.dumps(meta if meta is not None else valid_meta()))
    if landmarks is None:
        landmarks = [
            structs.LandmarkEntry("osm:node:1", 42.36, -71.05, "man_made"),
            structs.LandmarkEntry("osm:node:2", 42.35, -71.03, "natural"),
        ]
    (export_dir / "landmarks.json").write_bytes(
        msgspec.json.encode(landmarks, enc_hook=msgspec_enc_hook))
    if tables is None:
        tables = [structs.CompatibilityTable(
            tracklet_id="T1", matcher_version="m",
            entries=[structs.CompatibilityEntry("osm:node:1", 1.5)],
            default_log_lr=0.0, clip_lo=-4.0, clip_hi=4.0, status="fast")]
    (export_dir / "tier1_tables.json").write_bytes(
        msgspec.json.encode(tables, enc_hook=msgspec_enc_hook))
    if measurements is None:
        measurements = [structs.TrackletMeasurement(
            tracklet_id="T1", anchor_keyframe_idx=1,
            bearing_body_deg=45.0, kappa=100.0)]
    run_io.write_jsonl(export_dir / "tier1_measurements.jsonl", measurements)
    if odometry is None:
        odometry = [structs.OdometryDelta(
            keyframe_idx=k, forward_m=40.0, left_m=0.0, dyaw_rad=0.0,
            sigma_m=1.0, sigma_yaw_rad=0.02) for k in (1, 2)]
    run_io.write_jsonl(export_dir / "tier1_odometry.jsonl", odometry)
    run_io.write_jsonl(export_dir / "truth.jsonl", truth or [])
    return export_dir


class LoadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.export = Path(self.tmp.name) / "export"

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, **kwargs):
        kwargs.setdefault("max_visible_range_m", 10000.0)
        return export_ingest.load(self.export, **kwargs)

    def test_loads_a_valid_export(self):
        write_export(self.export)
        data = self.load()
        self.assertEqual(data.n_keyframes, 3)
        self.assertEqual(data.catalog.n, 2)
        self.assertEqual(len(data.measurements), 1)
        self.assertIn("214.0 deg", export_ingest.describe(data))
        # The catalog carries the radius the caller stated.
        self.assertEqual(float(data.catalog.max_visible_range_m[0]), 10000.0)

    def test_max_visible_range_is_required(self):
        write_export(self.export)
        with self.assertRaises(TypeError):
            export_ingest.load(self.export)

    def test_wrong_schema_version_is_refused(self):
        write_export(self.export, meta=valid_meta(schema_version="0.1"))
        with self.assertRaises(ValueError):
            self.load()

    def test_missing_mount_offset_provenance_is_refused(self):
        meta = valid_meta()
        del meta["mount_offset_deg"]
        del meta["mount_offset_source"]
        del meta["mount_offset_frame"]
        write_export(self.export, meta=meta)
        with self.assertRaises(msgspec.ValidationError):
            self.load()

    def test_wrong_offset_frame_is_refused(self):
        # The pohang shape: an offset reasoned in the column-0 frame.
        write_export(self.export,
                     meta=valid_meta(mount_offset_frame="column_0"))
        with self.assertRaises(ValueError) as ctx:
            self.load()
        self.assertIn("180", str(ctx.exception))

    def test_duplicate_information_epoch_is_refused(self):
        m = structs.TrackletMeasurement("T1", 1, 45.0, 100.0)
        write_export(self.export, measurements=[m, m])
        with self.assertRaises(ValueError) as ctx:
            self.load()
        self.assertIn("duplicate information epoch", str(ctx.exception))

    def test_measurement_without_table_is_refused(self):
        write_export(self.export, measurements=[
            structs.TrackletMeasurement("T_orphan", 1, 45.0, 100.0)])
        with self.assertRaises(ValueError) as ctx:
            self.load()
        self.assertIn("no table", str(ctx.exception))

    def test_table_scoring_unknown_landmark_is_refused(self):
        write_export(self.export, tables=[structs.CompatibilityTable(
            tracklet_id="T1", matcher_version="m",
            entries=[structs.CompatibilityEntry("osm:node:999", 1.0)],
            default_log_lr=0.0, clip_lo=-4.0, clip_hi=4.0, status="fast")])
        with self.assertRaises(ValueError) as ctx:
            self.load()
        self.assertIn("absent from the catalog", str(ctx.exception))

    def test_non_positive_kappa_is_refused(self):
        write_export(self.export, measurements=[
            structs.TrackletMeasurement("T1", 1, 45.0, 0.0)])
        with self.assertRaises(ValueError) as ctx:
            self.load()
        self.assertIn("kappa", str(ctx.exception))

    def test_out_of_range_bearing_is_refused(self):
        # Serialized bearings are [0, 360); a producer emitting wrap_deg
        # output violates the schema contract.
        write_export(self.export, measurements=[
            structs.TrackletMeasurement("T1", 1, -45.0, 100.0)])
        with self.assertRaises(ValueError) as ctx:
            self.load()
        self.assertIn("[0, 360)", str(ctx.exception))

    def test_noncontiguous_odometry_is_refused(self):
        write_export(self.export, odometry=[structs.OdometryDelta(
            keyframe_idx=2, forward_m=1.0, left_m=0.0, dyaw_rad=0.0,
            sigma_m=1.0, sigma_yaw_rad=0.02)])
        with self.assertRaises(ValueError) as ctx:
            self.load()
        self.assertIn("contiguous", str(ctx.exception))

    def test_region_box_spans_the_catalog(self):
        write_export(self.export)
        data = self.load()
        box = export_ingest.region_box(data, margin_m=100.0)
        self.assertLess(box.east_min_m, box.east_max_m)
        self.assertLess(box.north_min_m, box.north_max_m)


if __name__ == "__main__":
    unittest.main()
