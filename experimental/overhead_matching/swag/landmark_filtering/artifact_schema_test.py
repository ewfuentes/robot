import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    FilterPipelineConfig,
)


def make_test_artifact() -> schema.RunArtifact:
    obs = schema.Observation(
        obs_id="f0005__lm0__box0",
        pano_id="f0005",
        frame_idx=0,
        landmark_idx=0,
        embedding_id="f0005__landmark_0",
        primary_tag_key="building",
        primary_tag_value="university",
        additional_tags=[["name", "Green Building"]],
        confidence="high",
        description="Tall concrete building with radar domes",
        boxes=[schema.BBox(face_yaw_deg=0, xmin=509, ymin=289, xmax=606,
                           ymax=456)],
        seam_merged=False,
        bearing_camera_deg=5.2,
        bearing_global_deg=5.2,
        elevation_deg=3.0,
        angular_width_deg=8.5,
        decisions=[schema.FilterDecision(
            filter_name="angular_width_gate",
            disposition=schema.KEPT,
            reason="",
            details={"angular_width_deg": 8.5},
        )],
    )
    track = schema.Track(
        track_id=0,
        obs_ids=[obs.obs_id],
        first_frame_idx=0,
        last_frame_idx=0,
        representative_obs_id=obs.obs_id,
        triangulation=schema.TriangulationResult(
            solved=True,
            observability="near",
            x_m=10.0,
            y_m=20.0,
            lat=42.35,
            lon=-71.09,
            mean_range_m=25.0,
            residual_rms_deg=0.5,
            n_inliers=1,
            cov_enu=[[1.0, 0.1], [0.1, 2.0]],
            sigma_major_m=1.5,
            sigma_minor_m=0.9,
            parallax_deg=12.0,
        ),
    )
    return schema.RunArtifact(
        schema_version=schema.SCHEMA_VERSION,
        created_at="2026-07-15T00:00:00",
        git_hash="deadbeef",
        dataset_base="/data/x",
        pinhole_base="/data/y",
        landmark_base="/data/z",
        config=FilterPipelineConfig(),
        stages_run=["ingest"],
        anchor_lat=42.3544553,
        anchor_lon=-71.0912108,
        frames=[schema.Frame(
            frame_idx=0, pano_id="f0005",
            pano_stem="f0005,42.3544601,-71.0912099,",
            lat=42.3544601, lon=-71.0912099, x_m=0.0, y_m=0.0,
            dist_along_m=0.0, time_s=17.5, n_observations=1)],
        observations=[obs],
        tracks=[track],
        semantic_diagnostics=[schema.SemanticDiagnostics(
            backend="description_cosine",
            intra_track_similarity_histogram={"0.90": 3},
            top_cross_track_pairs=[schema.SimilarityPairExample(
                obs_id_a="a", obs_id_b="b", score=0.97, same_track=False)],
        )],
        stats=schema.SummaryStats(
            n_frames=1, n_observations=1, n_kept=1,
            filtered_by_reason={}, obs_per_frame_histogram={"1": 1}),
    )


class ArtifactSchemaTest(unittest.TestCase):
    def test_round_trip(self):
        artifact = make_test_artifact()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "run.json"
            schema.save_artifact(artifact, path)
            loaded = schema.load_artifact(path)
        self.assertEqual(artifact, loaded)

    def test_schema_version_present(self):
        artifact = make_test_artifact()
        self.assertEqual(artifact.schema_version, "1.0")

    def test_stage_a_only_artifact_is_valid(self):
        artifact = make_test_artifact()
        artifact.tracks = []
        artifact.semantic_diagnostics = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "run.json"
            schema.save_artifact(artifact, path)
            loaded = schema.load_artifact(path)
        self.assertEqual(loaded.tracks, [])


if __name__ == "__main__":
    unittest.main()
