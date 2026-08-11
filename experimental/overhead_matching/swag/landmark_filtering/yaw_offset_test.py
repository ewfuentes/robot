import unittest

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    yaw_offset,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    FilterPipelineConfig,
)


def build_synthetic_artifact(true_offset_deg: float,
                             true_drift_deg_per_frame: float = 0.0
                             ) -> schema.RunArtifact:
    """A walk east past three landmarks; camera bearings are compass bearings
    minus the (unknown, possibly drifting) offset."""
    rng = np.random.default_rng(7)
    n_frames = 25
    frames = [schema.Frame(
        frame_idx=i, pano_id=f"f{i:04d}", pano_stem=f"f{i:04d},42,-71,",
        lat=42.0, lon=-71.0, x_m=i * 5.0, y_m=0.0)
        for i in range(n_frames)]

    landmarks = [(60.0, 50.0), (40.0, -35.0), (90.0, 25.0)]
    observations = []
    tracks = []
    for lm_idx, (lx, ly) in enumerate(landmarks):
        obs_ids = []
        for i in range(n_frames):
            dx, dy = lx - frames[i].x_m, ly - frames[i].y_m
            compass = np.degrees(np.arctan2(dx, dy)) % 360.0
            camera = (compass - true_offset_deg
                      - true_drift_deg_per_frame * i
                      + rng.normal(0.0, 0.5)) % 360.0
            obs = schema.Observation(
                obs_id=f"f{i:04d}__lm{lm_idx}__box0", pano_id=f"f{i:04d}",
                frame_idx=i, landmark_idx=lm_idx, embedding_id=None,
                primary_tag_key="building", primary_tag_value="yes",
                additional_tags=[], confidence="high", description="",
                boxes=[schema.BBox(face_yaw_deg=0, xmin=450, ymin=450,
                                   xmax=550, ymax=550)],
                seam_merged=False, bearing_camera_deg=camera,
                bearing_global_deg=camera, elevation_deg=0.0,
                angular_width_deg=10.0, decisions=[], track_id=lm_idx)
            observations.append(obs)
            obs_ids.append(obs.obs_id)
        tracks.append(schema.Track(
            track_id=lm_idx, obs_ids=obs_ids, first_frame_idx=0,
            last_frame_idx=n_frames - 1, representative_obs_id=obs_ids[0]))

    return schema.RunArtifact(
        schema_version=schema.SCHEMA_VERSION, created_at="", git_hash="",
        dataset_base="", pinhole_base="", landmark_base="",
        config=FilterPipelineConfig(), stages_run=[], anchor_lat=42.0,
        anchor_lon=-71.0, frames=frames, observations=observations,
        tracks=tracks, stats=schema.SummaryStats())


def sweep_config(base_config):
    import msgspec
    return msgspec.structs.replace(
        base_config,
        yaw_offset=msgspec.structs.replace(
            base_config.yaw_offset, method="triangulation_sweep"))


class YawOffsetTest(unittest.TestCase):
    def test_recovers_known_offset(self):
        artifact = build_synthetic_artifact(true_offset_deg=37.0)
        offset, drift, method, details = yaw_offset.estimate_yaw_offset(
            artifact, sweep_config(artifact.config), None)
        self.assertEqual(method, "triangulation_sweep")
        self.assertAlmostEqual(offset, 37.0, delta=1.0)
        self.assertAlmostEqual(drift, 0.0, delta=0.02)
        # All three tracks triangulate cleanly at the optimum; the mirror
        # must have (near-)zero consensus.
        self.assertEqual(details["consensus"], 3.0)
        self.assertLess(details["mirror_consensus_at_plus_180"], 1.0)

    def test_recovers_linear_drift(self):
        artifact = build_synthetic_artifact(
            true_offset_deg=300.0, true_drift_deg_per_frame=-0.2)
        offset, drift, method, _ = yaw_offset.estimate_yaw_offset(
            artifact, sweep_config(artifact.config), None)
        self.assertAlmostEqual(offset, 300.0, delta=2.0)
        self.assertAlmostEqual(drift, -0.2, delta=0.03)

    def test_override_wins(self):
        artifact = build_synthetic_artifact(true_offset_deg=10.0)
        offset, drift, method, _ = yaw_offset.estimate_yaw_offset(
            artifact, artifact.config, 123.0)
        self.assertEqual(offset, 123.0)
        self.assertEqual(drift, 0.0)
        self.assertEqual(method, "fixed")

    def test_fixed_method(self):
        artifact = build_synthetic_artifact(true_offset_deg=10.0)
        offset, drift, method, _ = yaw_offset.estimate_yaw_offset(
            artifact, artifact.config, None)
        self.assertEqual(method, "fixed")
        self.assertEqual(offset, 0.0)


if __name__ == "__main__":
    unittest.main()
