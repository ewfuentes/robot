import unittest

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    semantic_similarity,
    tracking,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    AssociationConfig,
    FilterPipelineConfig,
)


def make_obs(frame_idx, landmark_idx, bearing, primary=("building", "yes"),
             confidence="high"):
    pano_id = f"f{frame_idx:04d}"
    return schema.Observation(
        obs_id=f"{pano_id}__lm{landmark_idx}__box0",
        pano_id=pano_id, frame_idx=frame_idx, landmark_idx=landmark_idx,
        embedding_id=None, primary_tag_key=primary[0],
        primary_tag_value=primary[1], additional_tags=[],
        confidence=confidence, description="",
        boxes=[schema.BBox(face_yaw_deg=0, xmin=450, ymin=450, xmax=550,
                           ymax=550)],
        seam_merged=False, bearing_camera_deg=bearing % 360.0,
        bearing_global_deg=bearing % 360.0, elevation_deg=0.0,
        angular_width_deg=10.0, decisions=[])


def make_artifact(observations, n_frames, frame_spacing_m=5.0,
                  association=None):
    frames = [schema.Frame(
        frame_idx=i, pano_id=f"f{i:04d}", pano_stem=f"f{i:04d},42,-71,",
        lat=42.0, lon=-71.0, x_m=0.0, y_m=i * frame_spacing_m)
        for i in range(n_frames)]
    config = FilterPipelineConfig()
    if association is not None:
        import msgspec
        config = msgspec.structs.replace(config, association=association)
    return schema.RunArtifact(
        schema_version=schema.SCHEMA_VERSION, created_at="", git_hash="",
        dataset_base="", pinhole_base="", landmark_base="",
        config=config, stages_run=["ingest"], anchor_lat=42.0,
        anchor_lon=-71.0, frames=frames, observations=observations,
        stats=schema.SummaryStats(n_observations=len(observations),
                                  n_kept=len(observations))), config


def run(observations, n_frames, association=None):
    artifact, config = make_artifact(observations, n_frames,
                                     association=association)
    tracking.run_tracking(
        artifact, config,
        backend=semantic_similarity.PrimaryTagEqualityBackend())
    return artifact


def tag_equality_association(**overrides):
    defaults = dict(semantic_backend="primary_tag_equality",
                    min_similarity=0.5)
    defaults.update(overrides)
    return AssociationConfig(**defaults)


class TrackingTest(unittest.TestCase):
    def test_single_stable_landmark_forms_one_track(self):
        obs = [make_obs(i, 0, bearing=90.0 + 0.5 * i) for i in range(6)]
        artifact = run(obs, 6, tag_equality_association())
        kept = [t for t in artifact.tracks if t.disposition == schema.KEPT]
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(kept[0].obs_ids), 6)
        self.assertTrue(all(o.track_id == kept[0].track_id for o in obs))

    def test_crossing_bearings_keep_identity(self):
        # Two landmarks with different tags whose bearings cross; the
        # semantic gate must keep identities apart even at the crossing.
        obs = []
        for i in range(8):
            obs.append(make_obs(i, 0, bearing=80.0 + 2.0 * i,
                                primary=("building", "office")))
            obs.append(make_obs(i, 1, bearing=96.0 - 2.0 * i,
                                primary=("tourism", "artwork")))
        artifact = run(obs, 8, tag_equality_association())
        kept = [t for t in artifact.tracks if t.disposition == schema.KEPT]
        self.assertEqual(len(kept), 2)
        for track in kept:
            tags = {artifact.observations[0].primary_tag_key}
            member_tags = {
                next(o for o in artifact.observations
                     if o.obs_id == oid).primary_tag_value
                for oid in track.obs_ids}
            self.assertEqual(len(member_tags), 1, member_tags)

    def test_out_of_gate_obs_spawns_new_track(self):
        obs = [make_obs(0, 0, bearing=90.0), make_obs(1, 0, bearing=90.5),
               make_obs(2, 0, bearing=91.0),
               # Same tag, far bearing: separate object.
               make_obs(2, 1, bearing=250.0), make_obs(3, 1, bearing=250.5),
               make_obs(4, 1, bearing=251.0)]
        artifact = run(obs, 5, tag_equality_association())
        kept = [t for t in artifact.tracks if t.disposition == schema.KEPT]
        self.assertEqual(len(kept), 2)

    def test_track_terminates_after_max_frame_gap(self):
        obs = ([make_obs(i, 0, bearing=90.0) for i in range(3)]
               + [make_obs(i, 0, bearing=90.0) for i in range(10, 13)])
        artifact = run(obs, 13,
                       tag_equality_association(max_frame_gap=3))
        kept = [t for t in artifact.tracks if t.disposition == schema.KEPT]
        self.assertEqual(len(kept), 2)

    def test_short_track_filtered_and_back_annotated(self):
        obs = [make_obs(0, 0, bearing=90.0), make_obs(1, 0, bearing=90.3)]
        artifact = run(obs, 2, tag_equality_association(min_track_length=3))
        self.assertEqual(len(artifact.tracks), 1)
        track = artifact.tracks[0]
        self.assertEqual(track.disposition, schema.FILTERED)
        self.assertEqual(track.reason, "track_too_short")
        for o in artifact.observations:
            self.assertEqual(o.final_disposition, schema.FILTERED)
            self.assertEqual(o.final_reason, "track_too_short")
        self.assertEqual(artifact.stats.n_tracks, 0)

    def test_filtered_observations_not_tracked(self):
        obs = [make_obs(i, 0, bearing=90.0) for i in range(4)]
        obs[1].final_disposition = schema.FILTERED
        obs[1].final_reason = "confidence_low"
        obs[1].decisions.append(schema.FilterDecision(
            filter_name="confidence_gate", disposition=schema.FILTERED,
            reason="confidence_low"))
        artifact = run(obs, 4, tag_equality_association())
        kept = [t for t in artifact.tracks if t.disposition == schema.KEPT]
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(kept[0].obs_ids), 3)
        self.assertIsNone(obs[1].track_id)

    def test_near_field_motion_gate_allowance(self):
        # A near landmark swings 8 deg/frame; with 5 m frame spacing and
        # default gain the widened gate should hold the track together.
        obs = [make_obs(i, 0, bearing=45.0 + 8.0 * i) for i in range(5)]
        artifact = run(obs, 5, tag_equality_association())
        kept = [t for t in artifact.tracks if t.disposition == schema.KEPT]
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(kept[0].obs_ids), 5)

    def test_singleton_count(self):
        obs = [make_obs(0, 0, bearing=90.0),
               make_obs(0, 1, bearing=200.0, primary=("shop", "bakery"))]
        artifact = run(obs, 1, tag_equality_association())
        self.assertEqual(artifact.stats.n_singleton_obs, 2)


class BackendInterfaceTest(unittest.TestCase):
    def test_tag_equality_pairwise(self):
        backend = semantic_similarity.PrimaryTagEqualityBackend()
        a = [make_obs(0, 0, 0.0, primary=("building", "yes")),
             make_obs(0, 1, 0.0, primary=("shop", "bakery"))]
        sim = backend.pairwise(a, a)
        np.testing.assert_allclose(sim, np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_observation_tags_dedup(self):
        obs = make_obs(0, 0, 0.0, primary=("building", "yes"))
        obs.additional_tags = [["name", "X"], ["building", "override"]]
        tags = semantic_similarity.observation_tags(obs)
        self.assertEqual(tags, {"building": "yes", "name": "X"})


if __name__ == "__main__":
    unittest.main()
