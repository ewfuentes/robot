import unittest
import types

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    causal_rigid_reranker,
    filter_catalog,
    structs,
)


def _mode(rank, probability, east_m, north_m):
    return {
        "source_rank": rank,
        "source_probability": probability,
        "east_m": east_m,
        "north_m": north_m,
        "heading_world_cw_deg": 0.0,
        "heading_index": 0,
        "north_index": rank,
        "east_index": rank,
    }


class CausalRigidRerankerTest(unittest.TestCase):

    def test_rigid_prefix_inverts_post_turn_left_motion(self):
        delta = structs.OdometryDelta(
            1, 10.0, 5.0, np.radians(70.0), 0.0, 0.0)

        paths = causal_rigid_reranker.reconstruct_rigid_prefixes(
            np.asarray([[13.0, 9.0, np.radians(90.0)]]), [delta], 1)

        np.testing.assert_allclose(
            paths[0],
            [[3.0, 4.0, np.radians(20.0)],
             [13.0, 9.0, np.radians(90.0)]],
            atol=1e-12)

    def test_rejects_noncausal_release_handoff(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            causal_rigid_reranker._release_maps([{
                "keyframe_idx": 1,
                "released_tracklet_ids": [],
                "returned": 1,
                "modes": [_mode(1, 1.0, 0.0, 0.0)],
            }], (), 2)

        catalog = filter_catalog.LandmarkCatalog(
            ["A"], [0.0], [100.0], max_visible_range_m=5000.0)
        release = types.SimpleNamespace(
            tracklet_id="A",
            measurements=(
                structs.TrackletMeasurement("A", -1, 0.0, 1000.0),),
            table=structs.CompatibilityTable(
                "A", "v", [structs.CompatibilityEntry("A", 4.0)],
                -4.0, -4.0, 4.0, "fast"))
        with self.assertRaisesRegex(ValueError, "invalid observation"):
            causal_rigid_reranker.score_rigid_prefixes(
                np.zeros((1, 2, 3)), (release,), catalog,
                pi0=0.2, matcher_recall=0.5, range_softness=0.25,
                range_cap=True, kappa_scale=1.0)

    def test_delayed_releases_use_no_future_and_never_rewrite(self):
        catalog = filter_catalog.LandmarkCatalog(
            ["A", "B", "background"], [0.0, 100.0, -1000.0],
            [100.0, 100.0, -1000.0], max_visible_range_m=5000.0)
        table_a = structs.CompatibilityTable(
            "A", "v", [structs.CompatibilityEntry("A", 4.0)],
            -4.0, -4.0, 4.0, "fast")
        table_b = structs.CompatibilityTable(
            "B", "v", [structs.CompatibilityEntry("B", 4.0)],
            -4.0, -4.0, 4.0, "fast")
        a = structs.TrackletMeasurement("A", 0, 0.0, 1000.0)
        b = tuple(structs.TrackletMeasurement(
            "B", keyframe, 0.0, 1000.0) for keyframe in (1, 2, 3))
        releases = (
            types.SimpleNamespace(
                release_keyframe_idx=2, tracklet_id="A",
                measurements=(a,), table=table_a),
            types.SimpleNamespace(
                release_keyframe_idx=4, tracklet_id="B",
                measurements=b, table=table_b),
        )
        odometry = [structs.OdometryDelta(
            keyframe, 10.0, 0.0, 0.0, 1.0, 0.01)
            for keyframe in range(1, 5)]
        snapshots = [
            {
                "keyframe_idx": 2,
                "released_tracklet_ids": ["A"],
                "returned": 2,
                # The appearance/geometry score must beat source probability.
                "modes": [
                    _mode(1, 0.99, 100.0, 20.0),
                    _mode(2, 0.01, 0.0, 20.0),
                ],
            },
            {
                "keyframe_idx": 4,
                "released_tracklet_ids": ["B"],
                "returned": 2,
                "modes": [
                    _mode(1, 0.99, 0.0, 40.0),
                    _mode(2, 0.01, 100.0, 40.0),
                ],
            },
        ]
        map_states = [{
            "east_m": -500.0,
            "north_m": float(10 * keyframe),
            "heading_world_cw_deg": 0.0,
        } for keyframe in range(5)]
        options = dict(
            pi0=0.2, matcher_recall=0.5, range_softness=0.25,
            range_cap=True, kappa_scale=1.0)

        estimates, selections = causal_rigid_reranker.causal_rerank(
            snapshots, map_states, releases, odometry, catalog, **options)
        self.assertEqual(
            [item["selected_source_rank"] for item in selections], [2, 2])
        # Anchor-0 evidence arrives at k=2: k=0/1 stay as already emitted.
        np.testing.assert_allclose(estimates[:2, :2], [
            [-500.0, 0.0], [-500.0, 10.0]])
        # The selected k=2 pose is propagated until the next atomic release.
        np.testing.assert_allclose(estimates[2:4, :2], [
            [0.0, 20.0], [0.0, 30.0]], atol=1e-12)

        # Altering unreleased future observations cannot affect the prefix.
        future_changed = tuple(structs.TrackletMeasurement(
            "B", keyframe, 180.0, 1000.0) for keyframe in (1, 2, 3))
        changed_releases = (
            releases[0],
            types.SimpleNamespace(
                release_keyframe_idx=4, tracklet_id="B",
                measurements=future_changed, table=table_b),
        )
        changed, _ = causal_rigid_reranker.causal_rerank(
            snapshots, map_states, changed_releases, odometry, catalog,
            **options)
        np.testing.assert_allclose(changed[:4], estimates[:4], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
