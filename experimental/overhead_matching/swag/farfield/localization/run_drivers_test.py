import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import msgspec

from common.python.serialization import msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    run_export,
    run_io,
    run_localization,
    structs,
)


def write_tiny_export(export_dir: Path) -> None:
    export_dir.mkdir(parents=True)
    (export_dir / "export_meta.json").write_text(json.dumps({
        "schema_version": structs.SCHEMA_VERSION,
        "scenario_name": "tiny",
        "anchor_lat_deg": 42.35, "anchor_lon_deg": -71.05,
        "n_keyframes": 6, "matcher_version": "m",
        "mount_offset_deg": 0.0, "mount_offset_source": "test",
        "mount_offset_frame": geo.MOUNT_OFFSET_FRAME}))
    landmarks = [structs.LandmarkEntry("osm:node:1", 42.36, -71.05, "x"),
                 structs.LandmarkEntry("osm:node:2", 42.35, -71.03, "y")]
    (export_dir / "landmarks.json").write_bytes(
        msgspec.json.encode(landmarks, enc_hook=msgspec_enc_hook))
    tables = [structs.CompatibilityTable(
        "T1", "m", [structs.CompatibilityEntry("osm:node:1", 1.0)],
        0.0, -4.0, 4.0, "fast")]
    (export_dir / "tier1_tables.json").write_bytes(
        msgspec.json.encode(tables, enc_hook=msgspec_enc_hook))
    run_io.write_jsonl(export_dir / "tier1_measurements.jsonl", [
        structs.TrackletMeasurement("T1", 2, 10.0, 100.0)])
    run_io.write_jsonl(export_dir / "tier1_odometry.jsonl", [
        structs.OdometryDelta(k, 40.0, 0.0, 0.0, 1.0, 0.05)
        for k in range(1, 6)])
    run_io.write_jsonl(export_dir / "truth.jsonl", [
        structs.TruthPose(k, 40.0 * k, 0.0, 90.0) for k in range(6)])


def export_argv(export_dir, out_dir, *extra):
    return ["run_export",
            "--export_dir", str(export_dir),
            "--output_dir", str(out_dir),
            "--init", "uniform",
            "--n_particles", "500",
            "--margin_m", "500",
            "--max_visible_range_m", "10000",
            "--position_roughening_m", "25",
            "--heading_roughening_deg", "1",
            "--checkpoint_every", "2",
            *extra]


class RunExportTest(unittest.TestCase):
    def test_uniform_run_records_provenance_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "export"
            out = Path(tmp) / "run"
            write_tiny_export(export_dir)
            with mock.patch("sys.argv",
                            export_argv(export_dir, out, "--seed", "3")):
                run_export.main()
            data = run_io.read_run(out)
        manifest = data.manifest
        self.assertEqual(manifest.export_dir, str(export_dir))
        self.assertEqual(manifest.max_visible_range_m, 10000.0)
        self.assertTrue(manifest.git_commit)
        self.assertIn("--seed", manifest.argv)
        self.assertEqual(len(data.measurements), 1)
        self.assertEqual(len(data.health), 6)

    def test_no_bearings_writes_the_run_that_happened(self):
        # The old driver wrote the FULL unconsumed measurements under
        # --no_bearings, so every odometry-only control on disk described
        # a run that never happened.
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "export"
            out = Path(tmp) / "run"
            write_tiny_export(export_dir)
            with mock.patch("sys.argv",
                            export_argv(export_dir, out, "--no_bearings")):
                run_export.main()
            data = run_io.read_run(out)
        self.assertEqual(data.measurements, [])
        self.assertEqual(data.tables, {})
        self.assertIn("bearings withheld", data.manifest.matcher_version)

    def test_truth_init_requires_prior_sigma(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "export"
            write_tiny_export(export_dir)
            argv = export_argv(export_dir, Path(tmp) / "run")
            argv[argv.index("uniform")] = "truth"
            with mock.patch("sys.argv", argv), \
                    self.assertRaises(SystemExit):
                run_export.main()


class RunLocalizationTest(unittest.TestCase):
    def test_synthetic_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            argv = ["run_localization",
                    "--scenario", "harbor_loop",
                    "--output_dir", str(out),
                    "--init", "local",
                    "--n_particles", "400",
                    "--max_visible_range_m", "10000",
                    "--prior_sigma_m", "300",
                    "--prior_offset_east_m", "100",
                    "--prior_offset_north_m", "-100",
                    "--keyframe_period_s", "2.0",
                    "--epoch_length", "5",
                    "--bearing_sigma_deg", "1.0"]
            with mock.patch("sys.argv", argv):
                run_localization.main()
            data = run_io.read_run(out)
        self.assertEqual(data.manifest.export_dir, "synthetic:harbor_loop")
        self.assertEqual(data.manifest.max_visible_range_m, 10000.0)
        self.assertTrue(data.manifest.git_commit)
        self.assertGreater(len(data.health), 10)

    def test_kidnap_requires_explicit_teleport(self):
        argv = ["run_localization", "--scenario", "harbor_loop",
                "--output_dir", "/tmp/x", "--init", "global",
                "--n_particles", "400", "--max_visible_range_m", "10000",
                "--box_halfwidth_m", "2500",
                "--keyframe_period_s", "2.0", "--epoch_length", "5",
                "--bearing_sigma_deg", "1.0", "--kidnap_at", "20"]
        with mock.patch("sys.argv", argv), self.assertRaises(SystemExit):
            run_localization.main()


if __name__ == "__main__":
    unittest.main()
