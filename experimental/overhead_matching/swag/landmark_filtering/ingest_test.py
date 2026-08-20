import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    ingest,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)


def make_landmark(boxes, name="Test Building", confidence="high"):
    return {
        "primary_tag": {"key": "building", "value": "yes"},
        "additional_tags": [{"key": "name", "value": name}],
        "confidence": confidence,
        "description": f"description of {name}",
        "bounding_boxes": boxes,
    }


def box(yaw, xmin, ymin, xmax, ymax):
    return {"yaw_angle": str(yaw), "xmin": xmin, "ymin": ymin, "xmax": xmax,
            "ymax": ymax}


def write_dataset(base: Path, pano_landmarks: dict[str, list],
                  gps_rows: list[str] | None = None):
    """pano_landmarks: {pano_stem: [landmark, ...]}"""
    pano_dir = base / "dataset" / "panorama"
    pano_dir.mkdir(parents=True)
    for stem in pano_landmarks:
        (pano_dir / f"{stem}.jpg").touch()
    if gps_rows is not None:
        header = ("idx,video_t_s,sensor_elapsed_s,dist_m,latitude,longitude,"
                  "altitude_m,speed_mps,gps_valid,frame_file")
        (base / "dataset" / "frames_gps.csv").write_text(
            "\n".join([header] + gps_rows) + "\n")

    results_dir = (base / "landmarks" / "sentences" / "results" / "req_000"
                   / "prediction-model-test")
    results_dir.mkdir(parents=True)
    lines = []
    for stem, landmarks in pano_landmarks.items():
        text = json.dumps({"location_type": "urban", "landmarks": landmarks})
        lines.append(json.dumps({
            "key": stem,
            "response": {"candidates": [{"content": {"parts": [
                {"text": text}]}}]},
        }))
    (results_dir / "predictions.jsonl").write_text("\n".join(lines) + "\n")
    return base / "dataset", base / "landmarks"


class IngestTest(unittest.TestCase):
    def run_ingest(self, pano_landmarks, gps_rows=None, config=None):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_base, landmark_base = write_dataset(
                Path(tmp), pano_landmarks, gps_rows)
            return ingest.run_ingest(
                dataset_base, landmark_base, config or IngestConfig())

    def test_seam_merge_across_faces(self):
        # Corrected 2026-08-19: the face image-right of face 0 is face 270, not
        # face 90. The panorama runs 180|90|0|270 left to right, so in camera
        # azimuth the faces are ordered 180 -> 90 -> 0 -> 270 and face 0's right
        # edge (az 45) is face 270's left edge. A box ending at face 0's right
        # edge and one starting at face 270's left edge are one object centred on
        # that 45 deg seam.
        result = self.run_ingest({
            "f0000,42.35,-71.09,": [make_landmark(
                [box(0, 800, 300, 1000, 500), box(270, 0, 310, 200, 510)])],
        })
        self.assertEqual(len(result.observations), 1)
        obs = result.observations[0]
        self.assertTrue(obs.seam_merged)
        self.assertEqual(len(obs.boxes), 2)
        self.assertAlmostEqual(obs.bearing_camera_deg, 45.0, delta=1.0)

    def test_seam_merge_requires_y_overlap(self):
        result = self.run_ingest({
            "f0000,42.35,-71.09,": [make_landmark(
                [box(0, 800, 0, 1000, 100), box(270, 0, 800, 200, 1000)])],
        })
        self.assertEqual(len(result.observations), 2)
        self.assertFalse(any(o.seam_merged for o in result.observations))

    def test_three_face_transitive_merge(self):
        result = self.run_ingest({
            "f0000,42.35,-71.09,": [make_landmark([
                box(90, 900, 300, 1000, 500),
                box(0, 0, 300, 1000, 500),
                box(270, 0, 300, 100, 500),
            ])],
        })
        self.assertEqual(len(result.observations), 1)
        obs = result.observations[0]
        self.assertEqual(len(obs.boxes), 3)
        # Corrected 2026-08-19: the azimuth-ordered chain is 90 -> 0 -> 270, so
        # this spans from inside face 90 (~306 deg) through face 0 to inside
        # face 270 (~50 deg), crossing both the 315 and 45 deg seams.
        self.assertGreater(obs.angular_width_deg, 90.0)
        self.assertAlmostEqual(obs.bearing_camera_deg, 0.0, delta=15.0)

    def test_wrap_seam_0_to_270(self):
        # Renamed and re-paired 2026-08-19: face 0's right edge adjoins face
        # 270's left edge (both sit on az 45). The old 270-then-0 pairing
        # followed the pre-fix `+90` rule.
        result = self.run_ingest({
            "f0000,42.35,-71.09,": [make_landmark(
                [box(0, 950, 300, 1000, 500), box(270, 0, 290, 60, 490)])],
        })
        self.assertEqual(len(result.observations), 1)
        obs = result.observations[0]
        self.assertTrue(obs.seam_merged)
        # Centred on the 45 deg seam, not 315: face 0 spans camera azimuth
        # 315..45 and face 270 spans 45..135, so their shared edge is 45.
        self.assertAlmostEqual(obs.bearing_camera_deg, 45.0, delta=2.0)

    def test_multi_instance_boxes_stay_separate(self):
        # Two boxes on opposite faces (e.g. "trees") are separate observations
        # sharing landmark_idx.
        result = self.run_ingest({
            "f0000,42.35,-71.09,": [make_landmark(
                [box(0, 100, 300, 300, 500), box(180, 100, 300, 300, 500)])],
        })
        self.assertEqual(len(result.observations), 2)
        self.assertEqual({o.landmark_idx for o in result.observations}, {0})
        self.assertEqual({o.obs_id for o in result.observations},
                         {"f0000__lm0__box0", "f0000__lm0__box1"})

    def test_gps_join_and_missing_rows(self):
        result = self.run_ingest(
            {
                "f0000,42.35,-71.09,": [make_landmark(
                    [box(0, 100, 300, 300, 500)])],
                "f0001,42.36,-71.09,": [],
            },
            gps_rows=[
                "0,0.0,-0.9,0.0,42.35,-71.09,-23.1,1.6,1,f0000_t0.jpg"],
        )
        self.assertEqual(len(result.frames), 2)
        self.assertEqual(result.frames[0].time_s, 0.0)
        self.assertIsNone(result.frames[1].time_s)

    def test_invalid_yaw_boxes_counted(self):
        result = self.run_ingest({
            "f0000,42.35,-71.09,": [make_landmark(
                [box(45, 100, 300, 300, 500)])],
        })
        self.assertEqual(len(result.observations), 0)
        self.assertEqual(result.stats.n_boxes_invalid_yaw, 1)
        self.assertEqual(result.stats.n_landmarks_without_valid_boxes, 1)

    def test_frame_enu_positions(self):
        result = self.run_ingest({
            "f0000,42.35,-71.09,": [],
            "f0001,42.36,-71.09,": [],
        })
        # Frames straddle the anchor north-south.
        self.assertLess(result.frames[0].y_m, 0.0)
        self.assertGreater(result.frames[1].y_m, 0.0)
        self.assertAlmostEqual(result.frames[0].x_m, 0.0, places=6)

    def test_observations_start_kept(self):
        result = self.run_ingest({
            "f0000,42.35,-71.09,": [make_landmark(
                [box(0, 100, 300, 300, 500)])],
        })
        obs = result.observations[0]
        self.assertEqual(obs.final_disposition, schema.KEPT)
        self.assertEqual(obs.decisions, [])
        self.assertEqual(result.stats.n_kept, 1)


if __name__ == "__main__":
    unittest.main()
