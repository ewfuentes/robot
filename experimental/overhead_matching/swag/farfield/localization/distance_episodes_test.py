import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import msgspec

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import (
    distance_episodes as episodes,
    export_ingest,
    release_schedule,
    structs,
)


class Meta(msgspec.Struct):
    n_keyframes: int
    prior_region: object


class DistanceEpisodesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.schedule = Path(self.tmp.name) / "schedule.json"
        self.schedule.write_text("bound release sidecar")
        self.truth = [structs.TruthPose(i, 100.0 * i, 0.0, 90.0)
                      for i in range(9)]
        self.ref = artifact.ArtifactRef(
            kind="localization_inputs", dataset="test", version="v1",
            content_digest="a" * 64, manifest_digest="b" * 64,
            path="/original/inputs")
        self.births = {"left": 0, "middle": 2, "crossing": 0, "eof": 4}
        self.releases = []
        for tid, close, anchors in (("left", 3, (2, 3)),
                                    ("middle", 5, (2, 5)),
                                    ("crossing", 5, (3, 4)),
                                    ("eof", 8, (4, 7))):
            table = structs.CompatibilityTable(tid, "test", [], -12, -12, 12, "fast")
            self.releases.append(release_schedule.TrackRelease(
                close, tid, tuple(structs.TrackletMeasurement(tid, i, 0.0, 1.0)
                                  for i in anchors), table))
        self.data = export_ingest.ExportData(
            self.ref, SimpleNamespace(), Meta(9, object()), object(), object(),
            [], [], [m for r in self.releases for m in r.measurements],
            {r.tracklet_id: r.table for r in self.releases}, self.truth)
        self.plan = {
            "schema": episodes.SCHEMA, "localization_inputs": self.ref.to_dict(),
            "release_schedule_sha256": artifact.sha256_file(self.schedule),
            "source_track_births": self.births,
            "windows": [[0, 4], [2, 6], [4, 8]],
            "boundary_policy": "fully_contained_source_track_inclusive_end",
        }

    def test_distance_windows_and_invalid_geometry(self):
        self.assertEqual(episodes.windows(self.truth, 3), [(0, 4), (2, 6), (4, 8)])
        self.assertEqual(episodes.windows(self.truth, 1), [(0, 8)])
        uneven = [structs.TruthPose(i, x, 0.0, 90.0)
                  for i, x in enumerate((0, 1, 2, 3, 200, 400, 600, 700, 800))]
        self.assertEqual(episodes.windows(uneven, 3), [(0, 5), (4, 6), (5, 8)])
        for count in (0, -1, True, 1.5, 100):
            with self.subTest(count=count), self.assertRaises(ValueError):
                episodes.windows(self.truth, count)
        for positions in ([0.0] * 9, [float('nan')] * 9):
            truth = [structs.TruthPose(i, x, 0.0, 0.0) for i, x in enumerate(positions)]
            with self.assertRaises(ValueError):
                episodes.windows(truth, 3)

    def test_five_equal_windows_and_separate_full_trajectory(self):
        truth = [structs.TruthPose(i, 100.0 * i, 0.0, 90.0)
                 for i in range(13)]
        self.assertEqual(episodes.windows(truth, 5),
                         [(0, 4), (2, 6), (4, 8), (6, 10), (8, 12)])
        self.assertEqual(episodes.windows(truth, 1), [(0, 12)])

    def test_whole_source_containment_and_reindexing_without_prior_crop(self):
        view, releases, meta = episodes.select(self.data, self.releases, self.plan, 1, self.schedule)
        self.assertEqual([r.tracklet_id for r in releases], ["middle"])
        self.assertEqual(releases[0].release_keyframe_idx, 3)
        self.assertEqual([m.anchor_keyframe_idx for m in releases[0].measurements], [0, 3])
        self.assertEqual([p.keyframe_idx for p in view.truth], list(range(5)))
        self.assertEqual(view.truth[0].east_m, 200.0)
        self.assertIs(view.catalog, self.data.catalog)
        self.assertIs(view.meta.prior_region, self.data.meta.prior_region)
        self.assertIs(view.artifact_ref, self.data.artifact_ref)
        self.assertEqual(meta['kept_tracklet_ids'], ['middle'])
        self.assertEqual(len(self.data.truth), 9)
        self.assertEqual(len(self.data.tables), 4)
        # A parent's EOF track is never flushed early at a synthetic cut.
        _, last, _ = episodes.select(self.data, self.releases, self.plan, 2, self.schedule)
        self.assertEqual([(r.tracklet_id, r.release_keyframe_idx) for r in last], [('eof', 4)])

    def test_trajectory_selection_does_not_require_tracks_or_schedule(self):
        plan = {**self.plan, "boundary_policy": "trajectory_only"}
        view, meta = episodes.select_trajectory(self.data, plan, 1)
        self.assertEqual([p.keyframe_idx for p in view.truth], list(range(5)))
        self.assertEqual(view.truth[0].east_m, 200.0)
        self.assertEqual(view.measurements, [])
        self.assertEqual(view.tables, {})
        self.assertEqual(meta["parent_keyframe_start"], 2)
        self.assertEqual(meta["boundary_policy"], "trajectory_only")

    def test_binding_and_birth_guards_and_relocation(self):
        relocated = copy.deepcopy(self.plan)
        relocated['localization_inputs']['path'] = '/another/machine/inputs'
        episodes.select(self.data, self.releases, relocated, 0, self.schedule)
        for field, value in [('schema', 'bad'), ('release_schedule_sha256', '0' * 64),
                             ('windows', [[0, 3], [2, 6], [4, 8]]),
                             ('source_track_births', {**self.births, 'left': True}),
                             ('source_track_births', {**self.births, 'left': 3})]:
            plan = {**self.plan, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                episodes.select(self.data, self.releases, plan, 0, self.schedule)
        for index in (-1, 3, True):
            with self.assertRaises(ValueError):
                episodes.select(self.data, self.releases, self.plan, index, self.schedule)


if __name__ == '__main__':
    unittest.main()
