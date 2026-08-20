import json
import math
import tempfile
import unittest
from pathlib import Path

import msgspec
import numpy as np

from common.python.serialization import msgspec_enc_hook
from experimental.overhead_matching.swag.farfield.localization import (
    gps_to_odometry,
    structs,
)


def derive(east, north, **kwargs):
    kwargs.setdefault("sigma_pair_m", 1.0)
    kwargs.setdefault("min_step_m", 2.0)
    return gps_to_odometry.derive_increments(east, north, **kwargs)


class DeriveIncrementsTest(unittest.TestCase):
    def test_straight_track_east(self):
        east = np.arange(5) * 40.0
        north = np.zeros(5)
        increments = derive(east, north)
        self.assertEqual([i.keyframe_idx for i in increments], [1, 2, 3, 4])
        for increment in increments:
            self.assertAlmostEqual(increment.forward_m, 40.0, places=9)
            self.assertEqual(increment.left_m, 0.0)
            self.assertEqual(increment.dyaw_rad, 0.0)
            self.assertEqual(increment.sigma_m, 1.0)
        # First step has no previous course: slow/gapped sigma. Later steps
        # carry the honest geometric budget hypot(atan(1/40), atan(1/40)).
        self.assertAlmostEqual(increments[0].sigma_yaw_rad,
                               math.radians(30.0), places=9)
        expected = math.hypot(math.atan(1.0 / 40.0), math.atan(1.0 / 40.0))
        for increment in increments[1:]:
            self.assertAlmostEqual(increment.sigma_yaw_rad, expected,
                                   places=9)

    def test_turn_dyaw_is_differenced_course(self):
        # Two steps east, then two steps north: one -90 deg course change.
        east = np.array([0.0, 30.0, 60.0, 60.0, 60.0])
        north = np.array([0.0, 0.0, 0.0, 30.0, 60.0])
        increments = derive(east, north)
        dyaws = [math.degrees(i.dyaw_rad) for i in increments]
        self.assertAlmostEqual(dyaws[0], 0.0, places=9)
        self.assertAlmostEqual(dyaws[1], 0.0, places=9)
        self.assertAlmostEqual(dyaws[2], -90.0, places=9)
        self.assertAlmostEqual(dyaws[3], 0.0, places=9)

    def test_speed_gate_and_gap_catch_up(self):
        # East, then two crawling steps (below min_step_m), then north:
        # the gated steps emit dyaw 0 at the slow sigma, and the catch-up
        # step spans the whole gap's course change.
        east = np.array([0.0, 30.0, 30.5, 31.0, 31.0])
        north = np.array([0.0, 0.0, 0.0, 0.0, 30.0])
        increments = derive(east, north)
        slow = math.radians(30.0)
        for gated in increments[1:3]:
            self.assertEqual(gated.dyaw_rad, 0.0)
            self.assertAlmostEqual(gated.sigma_yaw_rad, slow, places=9)
        self.assertAlmostEqual(math.degrees(increments[3].dyaw_rad), -90.0,
                               places=9)
        self.assertLess(increments[3].sigma_yaw_rad, slow)

    def test_declared_course_sigma_tracks_step_length(self):
        east = np.array([0.0, 50.0, 55.0])
        north = np.zeros(3)
        increments = derive(east, north)
        expected = math.hypot(math.atan(1.0 / 50.0), math.atan(1.0 / 5.0))
        self.assertAlmostEqual(increments[1].sigma_yaw_rad, expected,
                               places=9)

    def test_noise_injection_declares_itself(self):
        east = np.arange(20) * 40.0
        north = np.zeros(20)
        clean = derive(east, north)
        noisy = derive(east, north, extra_sigma_m=3.0,
                       extra_yaw_sigma_deg=2.0, noise_seed=7)
        self.assertNotEqual([i.forward_m for i in clean],
                            [i.forward_m for i in noisy])
        for a, b in zip(clean, noisy):
            self.assertAlmostEqual(b.sigma_m, math.hypot(a.sigma_m, 3.0),
                                   places=9)
            self.assertAlmostEqual(
                b.sigma_yaw_rad,
                math.hypot(a.sigma_yaw_rad, math.radians(2.0)), places=9)

    def test_deterministic(self):
        east = np.arange(10) * 40.0
        north = np.linspace(0.0, 90.0, 10)
        a = derive(east, north, extra_sigma_m=1.0)
        b = derive(east, north, extra_sigma_m=1.0)
        self.assertEqual(a, b)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            derive([0.0], [0.0])
        with self.assertRaises(ValueError):
            derive([0.0, 1.0], [0.0, 1.0], sigma_pair_m=0.0)

    def test_modeling_knobs_are_required(self):
        with self.assertRaises(TypeError):
            gps_to_odometry.derive_increments([0.0, 40.0], [0.0, 0.0])


class RewriteExportTest(unittest.TestCase):
    def _make_export(self, root: Path) -> Path:
        export_dir = root / "export"
        export_dir.mkdir()
        truth = [structs.TruthPose(keyframe_idx=k, east_m=40.0 * k,
                                   north_m=0.0, heading_deg=90.0)
                 for k in range(6)]
        with open(export_dir / "truth.jsonl", "wb") as f:
            for pose in truth:
                f.write(msgspec.json.encode(pose,
                                            enc_hook=msgspec_enc_hook))
                f.write(b"\n")
        (export_dir / "export_meta.json").write_text(json.dumps(
            {"schema_version": "0.2", "scenario_name": "tiny",
             "anchor_lat_deg": 42.0, "anchor_lon_deg": -71.0,
             "n_keyframes": 6, "matcher_version": "m",
             "custom_field_this_build_ignores": True}))
        (export_dir / "landmarks.json").write_text("[]")
        (export_dir / "tier1_tables.json").write_text("[]")
        (export_dir / "tier1_measurements.jsonl").write_text("")
        return export_dir

    def test_rewrite_regenerates_odometry_and_records_derivation(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = self._make_export(Path(tmp))
            out = Path(tmp) / "out"
            increments = gps_to_odometry.rewrite_export(
                export_dir, out, sigma_pair_m=1.0, min_step_m=2.0)
            self.assertEqual(len(increments), 5)
            meta = json.loads((out / "export_meta.json").read_text())
            self.assertEqual(meta["schema_version"], structs.SCHEMA_VERSION)
            self.assertEqual(meta["odometry_derivation"]["sigma_pair_m"], 1.0)
            # Unknown meta fields survive the round trip.
            self.assertTrue(meta["custom_field_this_build_ignores"])
            # Source untouched.
            src_meta = json.loads((export_dir / "export_meta.json").read_text())
            self.assertEqual(src_meta["schema_version"], "0.2")

    def test_refuses_in_place_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_dir = self._make_export(Path(tmp))
            with self.assertRaises(ValueError):
                gps_to_odometry.rewrite_export(
                    export_dir, export_dir, sigma_pair_m=1.0, min_step_m=2.0)


if __name__ == "__main__":
    unittest.main()
