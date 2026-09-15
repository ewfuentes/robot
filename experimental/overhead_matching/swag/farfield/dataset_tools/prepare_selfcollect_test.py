import csv
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.dataset_tools import (
    prepare_selfcollect as prepare,
)


class PrepareSelfcollectTest(unittest.TestCase):

    def test_load_timestamped_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gps.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["timestamp", "lat", "lon"])
                writer.writeheader()
                writer.writerows([
                    {"timestamp": "1700000000.25", "lat": "42.0", "lon": "-71.0"},
                    {"timestamp": "1700000001.25", "lat": "42.1", "lon": "-70.9"},
                ])
            rows, metadata = prepare.load_timestamped_csv(path)
            self.assertEqual([row["elapsed_s"] for row in rows], [0.0, 1.0])
            self.assertEqual(metadata["type"], "timestamped_csv")
            self.assertEqual(metadata["positioned_records"], 2)

    def test_data_root_paths_are_portable(self):
        self.assertEqual(prepare.data_root_relative(Path(
            "/data/farfield_matching/raw_material/collect/video.mp4")),
                         "raw_material/collect/video.mp4")

    def make_track(self):
        # Roughly 10 m east per second at the equator.
        rows = [{
            "elapsed_s": float(index),
            "latitude": 0.0,
            "longitude": index * 10.0 / prepare.EARTH_RADIUS_M * 180.0 / 3.141592653589793,
            "altitude_m": 100.0,
            "speed_mps": 10.0,
        } for index in range(31)]
        return prepare.Track(rows, sigma_s=0.0, velocity_limit_mps=20.0)

    def test_distance_grid_maps_to_output_frames(self):
        recording = {
            "dataset_id": "test_drive",
            "capture_fps": 30,
            "output_fps": 3,
            "sync": {"sensor_elapsed_at_video_start_s": 0.0},
            "trim_head_s": 1.0,
            "trim_tail_s": 1.0,
            "sampling": {"distance_m": 20.0, "course_radius_m": 20.0},
            "gps_quality": {"max_gap_s": 3.0, "fix_near_s": 1.5},
        }
        source_info = {
            "media_fps": 30.0,
            "nb_frames": 900,
            "duration_s": 30.0,
        }
        rows = prepare.sample_recording(
            self.make_track(), recording, source_info)
        self.assertGreater(len(rows), 10)
        self.assertEqual(len({row["frame_index"] for row in rows}), len(rows))
        self.assertTrue(all(row["source_capture_fps"] == 30 for row in rows))
        self.assertLessEqual(
            max(abs(row["frame_time_error_s"]) for row in rows), 1 / 6 + 1e-9)

    def test_too_fine_spacing_fails_instead_of_reusing_an_image(self):
        recording = {
            "dataset_id": "test_drive",
            "capture_fps": 30,
            "output_fps": 3,
            "sync": {"sensor_elapsed_at_video_start_s": 0.0},
            "sampling": {"distance_m": 1.0, "course_radius_m": 10.0},
            "gps_quality": {"max_gap_s": 3.0},
        }
        source_info = {
            "media_fps": 3.0,
            "nb_frames": 90,
            "duration_s": 30.0,
        }
        with self.assertRaisesRegex(ValueError, "too fine"):
            prepare.sample_recording(self.make_track(), recording, source_info)

    def test_source_clip_offsets_gps_but_keeps_output_time_relative(self):
        recording = {
            "dataset_id": "test_drive",
            "capture_fps": 30,
            "output_fps": 3,
            "clip_start_s": 5.0,
            "clip_end_s": 20.0,
            "sync": {"sensor_elapsed_at_video_start_s": 0.0},
            "sampling": {"distance_m": 20.0, "course_radius_m": 20.0},
            "gps_quality": {"max_gap_s": 3.0, "fix_near_s": 1.5},
        }
        source_info = {
            "media_fps": 30.0,
            "nb_frames": 900,
            "duration_s": 30.0,
        }
        rows = prepare.sample_recording(
            self.make_track(), recording, source_info)
        self.assertGreater(len(rows), 1)
        for row in rows:
            self.assertAlmostEqual(
                row["source_video_t_s"], row["video_t_s"] + 5.0)
            self.assertAlmostEqual(
                row["sensor_elapsed_s"], row["source_video_t_s"])
            self.assertGreaterEqual(row["frame_index"], 0)
            self.assertLess(row["frame_index"], 45)

    def test_extracted_frames_are_bound_by_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.jpg"
            frame.write_bytes(b"blurred-frame")
            expected = {frame.name: prepare.sha256_file(frame)}
            prepare.verify_extracted_frame_hashes(
                [frame], {frame.name}, expected, "test")
            frame.write_bytes(b"replaced-frame")
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare.verify_extracted_frame_hashes(
                    [frame], {frame.name}, expected, "test")


if __name__ == "__main__":
    unittest.main()
