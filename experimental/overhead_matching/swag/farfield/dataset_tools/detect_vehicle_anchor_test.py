import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.dataset_tools import (
    detect_vehicle_anchor as subject,
)


class ResolveFramePathTest(unittest.TestCase):

    def test_missing_named_frame_never_uses_sorted_positional_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            panorama = dataset / "panorama"
            panorama.mkdir()
            (panorama / "some_other_frame.jpg").write_bytes(b"not this row")
            rows = [{"frame_file": "recorded_frame.jpg"}]
            with self.assertRaisesRegex(FileNotFoundError, "recorded_frame"):
                subject.resolve_frame_path(dataset, rows, 0)

    def test_recorded_name_may_live_in_frames_or_panorama(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            frames = dataset / "frames"
            frames.mkdir()
            expected = frames / "recorded.jpg"
            expected.write_bytes(b"image")
            self.assertEqual(
                subject.resolve_frame_path(
                    dataset, [{"frame_file": "recorded.jpg"}], 0),
                expected)


if __name__ == "__main__":
    unittest.main()
