import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    extract_plan_frames as epf,
)


def fake_frames(count):
    for i in range(count):
        yield i, np.full((16, 32, 3), (i * 10, 20, 200 - i * 10), np.uint8)


class ExtractPlanFramesTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = self.root / "frames_gps.csv"
        self.plan.write_text(
            "idx,frame_index,video_t_s,frame_file\n"
            "0,2,0.667,leg_p000002_t000000.667s.jpg\n"
            "1,5,1.667,leg_p000005_t000001.667s.jpg\n"
            "2,9,3.000,leg_p000009_t000003.000s.jpg\n")

    def test_writes_wanted_frames_and_hashes(self):
        wanted = epf.read_plan(self.plan)
        self.assertEqual(sorted(wanted), [2, 5, 9])
        frames_dir = self.root / "frames"
        digests = epf.extract(fake_frames(50), wanted, frames_dir, 92)
        self.assertEqual(sorted(digests), sorted(wanted.values()))
        self.assertEqual(len(list(frames_dir.iterdir())), 3)
        for name, digest in digests.items():
            self.assertEqual(artifact.sha256_file(frames_dir / name), digest)
        img = cv2.imread(str(frames_dir / wanted[5]))
        self.assertLess(abs(int(img[8, 16, 0]) - 50), 6)        # frame 5's blue channel

    def test_short_video_fails_closed(self):
        wanted = epf.read_plan(self.plan)
        with self.assertRaises(RuntimeError):
            epf.extract(fake_frames(7), wanted, self.root / "frames", 92)

    def test_duplicate_plan_rows_rejected(self):
        self.plan.write_text("frame_index,frame_file\n1,a.jpg\n1,b.jpg\n")
        with self.assertRaises(ValueError):
            epf.read_plan(self.plan)


if __name__ == "__main__":
    unittest.main()
