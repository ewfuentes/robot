"""Tests for stream discovery against synthetic log trees.

Builds directory trees rather than using real data: what these exercise is which items a log is
judged to have and what happens when one is missing, and that logic is about paths, not about
parsing feather. The parsers belong to the `av2` devkit and are its tests' problem.
"""

import tempfile
import unittest
from pathlib import Path

from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.viz import av2_source


def _make_log(root: Path, request: al.Request, log_id: str, items) -> Path:
    """Create a log directory containing exactly `items`, empty but present."""
    log_dir = al.log_dir(request, log_id, root)
    log_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        path = al.local_path(request, log_id, item, root)
        if item.is_dir:
            path.mkdir(parents=True, exist_ok=True)
            # A directory item counts only when non-empty, so give it something.
            (path / "placeholder").write_bytes(b"")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
    return log_dir


class DiscoverLogIdsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.request = al.SensorRequest(split=al.SensorSplit.VAL)

    def test_missing_split_dir_is_empty_not_an_error(self):
        """Nothing downloaded yet is a normal state, not a failure."""
        self.assertEqual(av2_source.discover_log_ids(self.request, self.root), [])

    def test_finds_log_dirs_sorted(self):
        for log_id in ["ccc", "aaa", "bbb"]:
            _make_log(self.root, self.request, log_id, [al.SensorItem.POSES])
        self.assertEqual(
            av2_source.discover_log_ids(self.request, self.root), ["aaa", "bbb", "ccc"]
        )

    def test_ignores_stray_files(self):
        _make_log(self.root, self.request, "aaa", [al.SensorItem.POSES])
        (self.request.local_dir(self.root) / "catalog.json").write_bytes(b"{}")
        self.assertEqual(av2_source.discover_log_ids(self.request, self.root), ["aaa"])


class LogSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.request = al.SensorRequest(split=al.SensorSplit.VAL)

    def test_motion_forecasting_is_rejected_up_front(self):
        """Its scenarios are one parquet file, not a log dir; fail before touching disk."""
        request = al.MotionForecastingRequest(split=al.MotionForecastingSplit.VAL)
        with self.assertRaises(av2_source.UnsupportedDatasetError):
            av2_source.LogSource(request, "any", self.root)

    def test_missing_log_dir_names_the_path(self):
        with self.assertRaises(av2_source.MissingStreamError) as ctx:
            av2_source.LogSource(self.request, "nope", self.root)
        self.assertIn("nope", str(ctx.exception))

    def test_present_items_reports_only_what_is_on_disk(self):
        _make_log(self.root, self.request, "aaa",
                  [al.SensorItem.POSES, al.SensorItem.MAP, al.SensorItem.LIDAR])
        source = av2_source.LogSource(self.request, "aaa", self.root)
        self.assertEqual(
            set(source.present_items()),
            {al.SensorItem.POSES, al.SensorItem.MAP, al.SensorItem.LIDAR},
        )

    def test_empty_directory_item_does_not_count_as_present(self):
        """A `cp` that created the directory and then failed is not a downloaded stream."""
        _make_log(self.root, self.request, "aaa", [al.SensorItem.POSES])
        al.local_path(self.request, "aaa", al.SensorItem.LIDAR, self.root).mkdir(parents=True)
        source = av2_source.LogSource(self.request, "aaa", self.root)
        self.assertFalse(source.has(al.SensorItem.LIDAR))

    def test_missing_poses_error_lists_what_is_present(self):
        _make_log(self.root, self.request, "aaa", [al.SensorItem.MAP])
        source = av2_source.LogSource(self.request, "aaa", self.root)
        with self.assertRaises(av2_source.MissingStreamError) as ctx:
            source.city_SE3_ego()
        self.assertIn("map", str(ctx.exception))

    def test_stream_absent_from_the_dataset_says_so(self):
        """TBV ships no annotations at all, which is a different error from 'not downloaded'."""
        request = al.TbvRequest()
        _make_log(self.root, request, "aaa__Spring_2020", [al.TbvItem.POSES])
        source = av2_source.LogSource(request, "aaa__Spring_2020", self.root)
        with self.assertRaises(av2_source.MissingStreamError) as ctx:
            source._require_named("ANNOTATIONS")
        self.assertIn("tbv dataset has no annotations", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
