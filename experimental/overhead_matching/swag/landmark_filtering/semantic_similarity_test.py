import unittest

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    semantic_similarity,
)


def make_obs(obs_id, track_id, primary=("building", "yes"),
             disposition=schema.KEPT):
    return schema.Observation(
        obs_id=obs_id, pano_id=obs_id.split("__")[0], frame_idx=0,
        landmark_idx=0, embedding_id=None, primary_tag_key=primary[0],
        primary_tag_value=primary[1], additional_tags=[], confidence="high",
        description="", boxes=[schema.BBox(
            face_yaw_deg=0, xmin=0, ymin=0, xmax=100, ymax=100)],
        seam_merged=False, bearing_camera_deg=0.0, bearing_global_deg=0.0,
        elevation_deg=0.0, angular_width_deg=10.0, decisions=[],
        final_disposition=disposition, track_id=track_id)


class FixedBackend:
    """Similarity 1.0 for equal primary tags, else a fixed cross value."""

    name = "fixed"

    def __init__(self, cross=0.2):
        self.cross = cross

    def pairwise(self, obs_a, obs_b):
        out = np.zeros((len(obs_a), len(obs_b)))
        for i, a in enumerate(obs_a):
            for j, b in enumerate(obs_b):
                same = (a.primary_tag_key, a.primary_tag_value) == (
                    b.primary_tag_key, b.primary_tag_value)
                out[i, j] = 1.0 if same else self.cross
        return out


class DiagnosticsTest(unittest.TestCase):
    def test_intra_inter_separation(self):
        obs = [
            make_obs("f0000__lm0__box0", 0, ("building", "a")),
            make_obs("f0001__lm0__box0", 0, ("building", "a")),
            make_obs("f0000__lm1__box0", 1, ("shop", "b")),
            make_obs("f0001__lm1__box0", 1, ("shop", "b")),
        ]
        diag = semantic_similarity.compute_diagnostics(
            obs, FixedBackend(), num_example_pairs=10, max_obs=100)
        # Intra-track pairs all 1.0; inter-track all 0.2.
        self.assertEqual(diag.intra_track_similarity_histogram, {"1.00": 2})
        self.assertEqual(diag.inter_track_similarity_histogram, {"0.20": 4})
        self.assertTrue(
            all(not p.same_track for p in diag.top_cross_track_pairs))
        self.assertTrue(
            all(p.same_track for p in diag.bottom_intra_track_pairs))

    def test_untracked_and_filtered_obs_excluded(self):
        obs = [
            make_obs("f0000__lm0__box0", 0),
            make_obs("f0001__lm0__box0", 0),
            make_obs("f0002__lm0__box0", None),
            make_obs("f0003__lm0__box0", 1, disposition=schema.FILTERED),
        ]
        diag = semantic_similarity.compute_diagnostics(
            obs, FixedBackend(), num_example_pairs=10, max_obs=100)
        total = (sum(diag.intra_track_similarity_histogram.values())
                 + sum(diag.inter_track_similarity_histogram.values()))
        self.assertEqual(total, 1)  # only the one tracked kept pair

    def test_backend_agreement(self):
        obs = [
            make_obs("f0000__lm0__box0", 0, ("building", "a")),
            make_obs("f0001__lm0__box0", 0, ("building", "a")),
            make_obs("f0000__lm1__box0", 1, ("shop", "b")),
        ]
        agreement = semantic_similarity.compute_backend_agreement(
            obs, FixedBackend(cross=0.2), FixedBackend(cross=0.2),
            num_example_pairs=5, max_obs=100)
        self.assertEqual(agreement.n_pairs, 3)
        self.assertAlmostEqual(agreement.correlation, 1.0)
        self.assertEqual(agreement.n_large_disagreements, 0)

    def test_missing_embedding_error_message(self):
        err = semantic_similarity.MissingTextEmbeddingsError(
            ["Green Building", "Prudential Tower"], None)
        self.assertIn("Green Building", str(err))
        self.assertIn("precompute_value_embeddings", str(err))


if __name__ == "__main__":
    unittest.main()
