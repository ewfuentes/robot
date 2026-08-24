"""Tests for the log index, against the real log snippet.

Unlike `av2_log_test`, which builds synthetic empty trees because what it checks is path logic,
these read actual data. That is the point: the snippet is a genuine `tbv` log laid out exactly
as the downloader writes it, so a dataset that resolves paths correctly here resolves them
correctly against `/data`, and the devkit parsers get exercised on the way through.
"""

import unittest
from pathlib import Path

from experimental.map_estimation.data import argoverse_dataset as ad
from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.data import av2_log

# The snippet mirrors S3, so it is a dataset *root* -- the log sits at `tbv/<log_id>` under it,
# not directly inside it.
BASE_PATH = Path("external/argoverse_snippet")
LOG_ID = "07YOTznatmYypvQYpzviEcU3yGPsyaGg__Spring_2020"
LOG_IDS = [
    LOG_ID,
    "i1qcWZ15fSD2vfLljK8EgVPdyUWNgbp9__Winter_2021",
]


class ArgoverseDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.request = al.TbvRequest()

    def test_discovers_the_log_on_disk(self):
        dataset = ad.ArgoverseDataset(self.request, root=BASE_PATH)

        self.assertEqual(dataset.get_log_ids(), LOG_IDS)
        self.assertEqual(len(dataset), len(LOG_IDS))

    def test_named_log_ids_are_used_verbatim(self):
        dataset = ad.ArgoverseDataset(self.request, log_ids=[LOG_ID], root=BASE_PATH)

        self.assertEqual(dataset.get_log_ids(), [LOG_ID])

    def test_a_named_log_that_is_absent_raises(self):
        """A typo in a curated log list is an error, not a shorter dataset."""
        with self.assertRaises(av2_log.MissingStreamError):
            ad.ArgoverseDataset(self.request, log_ids=["not-a-log"], root=BASE_PATH)

    def test_reports_the_streams_the_snippet_ships(self):
        source = ad.ArgoverseDataset(self.request, root=BASE_PATH).log(LOG_ID)

        present = {item.token for item in source.present_items()}
        self.assertEqual(
            present,
            {"map", "calibration", "poses", "lidar",
             "ring_front_center", "ring_front_left", "ring_front_right",
             "ring_side_left", "ring_side_right", "ring_rear_left", "ring_rear_right"},
        )

    def test_streams_parse(self):
        """The streams are readable, not merely present -- this is what paths being right buys."""
        source = ad.ArgoverseDataset(self.request, root=BASE_PATH).log(LOG_ID)

        self.assertGreater(len(source.static_map().vector_lane_segments), 0)
        self.assertGreater(len(source.city_SE3_ego()), 0)
        self.assertEqual(len(list(source.lidar_sweeps())), 15)

    def test_lidar_and_camera_timestamps_are_distinct_streams(self):
        """Lidar runs at 10 Hz and the cameras at 20 Hz, so neither can stand in for the other.

        Guards the shape of bug that a hand-rolled loader invites: deriving one stream's
        timestamps from another's directory and getting a plausible-looking list back.
        """
        source = ad.ArgoverseDataset(self.request, root=BASE_PATH).log(LOG_ID)

        sweeps = [sweep.timestamp_ns for sweep in source.lidar_sweeps()]
        frames = [ts for ts, _ in source.camera_frames(al.TbvItem.RING_FRONT_CENTER)]

        self.assertEqual(len(sweeps), 15)
        # A range, not an exact count: the snippet is a time slice, so how many 20 Hz frames land
        # inside it depends on where that camera's phase falls -- the seven cameras hold 30 or 31
        # each off the same window. What is being asserted is the 2x rate, not the cut.
        self.assertIn(len(frames), (2 * len(sweeps), 2 * len(sweeps) + 1))
        self.assertEqual(set(sweeps) & set(frames), set())


if __name__ == "__main__":
    unittest.main()
