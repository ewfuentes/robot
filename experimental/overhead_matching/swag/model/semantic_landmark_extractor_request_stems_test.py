"""Request building must not pay for stems the dataset no longer contains.

Regression test for the 2026-08-17 folkestone_dover overspend: the pinhole
render artifact held 399 pre-trim stems while the dataset's panorama/ had been
trimmed to 105, and `create_panorama_sentences` built (and billed) a Gemini
request for every rendered stem. With `--panorama_dir` the request set is
restricted to the dataset's current panoramas, and a current panorama with no
render is a hard error rather than a silently absent frame.
"""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image

from experimental.overhead_matching.swag.model import (
    semantic_landmark_extractor as sle,
)

STEMS = ["f0000,42.0000000,-71.0000000,",
         "f0001,42.0010000,-71.0010000,",
         "f0002,42.0020000,-71.0020000,"]


def _write_jpg(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 90, 60)).save(path)


def _make_args(pinhole_dir, output_base, panorama_dir=None):
    return Namespace(
        pinhole_dir=str(pinhole_dir),
        panorama_dir=str(panorama_dir) if panorama_dir else None,
        output_base=str(output_base),
        prompt_type="osm_tags_farfield",
        num_workers=1,
        max_requests_per_batch=10000,
        disable_tqdm=True,
        pano_ids_file=None,
        max_panoramas=None,
        media_resolution="MEDIA_RESOLUTION_ULTRA_HIGH",
        thinking_level="HIGH",
    )


class RequestStemFilterTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        # Pinhole render carries all three stems; the dataset was trimmed to
        # the first two, so one rendered stem is stale.
        self.pinhole_dir = root / "pinhole"
        for stem in STEMS:
            for yaw in (0, 90, 180, 270):
                _write_jpg(self.pinhole_dir / stem / f"yaw_{yaw:03d}.jpg")
        self.panorama_dir = root / "panorama"
        for stem in STEMS[:2]:
            _write_jpg(self.panorama_dir / f"{stem}.jpg")
        self.output_base = root / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def _request_keys(self):
        request_dir = self.output_base / "panorama_sentence_requests"
        keys = []
        for batch_file in sorted(request_dir.glob("*.jsonl")):
            for line in batch_file.read_text().splitlines():
                keys.append(json.loads(line)["key"])
        return keys

    def test_requests_restricted_to_current_panoramas(self):
        sle.create_panorama_description_requests(
            _make_args(self.pinhole_dir, self.output_base, self.panorama_dir))

        self.assertEqual(sorted(self._request_keys()), sorted(STEMS[:2]))

    def test_without_panorama_dir_all_rendered_stems_are_requested(self):
        sle.create_panorama_description_requests(
            _make_args(self.pinhole_dir, self.output_base))

        self.assertEqual(sorted(self._request_keys()), sorted(STEMS))

    def test_current_panorama_without_render_is_an_error(self):
        _write_jpg(self.panorama_dir / "f0003,42.0030000,-71.0030000,.jpg")

        with self.assertRaisesRegex(RuntimeError, "no pinhole render"):
            sle.create_panorama_description_requests(
                _make_args(self.pinhole_dir, self.output_base,
                           self.panorama_dir))


if __name__ == "__main__":
    unittest.main()
