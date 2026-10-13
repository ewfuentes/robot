import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import common.torch.load_torch_deps  # noqa: F401
import pandas as pd
import torch

from experimental.overhead_matching.swag.scripts import export_ct2l_similarity as ex
from experimental.overhead_matching.swag.scripts import train_ct2l


def fake_dataset(root: Path, names: list[str]):
    return SimpleNamespace(_satellite_metadata=pd.DataFrame({"path": [str(root / "satellite" / n) for n in names]}))


class LoadCaptionsTest(unittest.TestCase):
    def test_accepts_pipeline_records_and_plain_strings(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"a": {"description": "desc a", "legible_text": []}, "b": "desc b"}))
            self.assertEqual(ex.load_captions(p), {"a": "desc a", "b": "desc b"})


class GalleryImagePathsTest(unittest.TestCase):
    def test_satellite_subdir_uses_metadata_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "satellite").mkdir()
            (root / "satellite" / "t.jpg").touch()
            ds = fake_dataset(root, ["t.jpg"])
            self.assertEqual(ex.gallery_image_paths(ds, "satellite"), [str(root / "satellite" / "t.jpg")])

    def test_other_subdir_same_stem_png_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "satellite_osm_carto").mkdir()
            (root / "satellite_osm_carto" / "t.png").touch()
            ds = fake_dataset(root, ["t.jpg"])
            self.assertEqual(ex.gallery_image_paths(ds, "satellite_osm_carto"),
                             [str(root / "satellite_osm_carto" / "t.png")])

    def test_missing_gallery_image_raises(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "satellite_osm_carto").mkdir()
            with self.assertRaises(FileNotFoundError):
                ex.gallery_image_paths(fake_dataset(root, ["t.jpg"]), "satellite_osm_carto")


class SimilarityWithZeroRowsTest(unittest.TestCase):
    def test_uncaptioned_rows_are_zero(self):
        captioned = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        out = ex.similarity_with_zero_rows(captioned, [0, 2], num_panos=3)
        torch.testing.assert_close(out, torch.tensor([[0.1, 0.2], [0.0, 0.0], [0.3, 0.4]]))


class BuildPairsTest(unittest.TestCase):
    def test_skips_uncaptioned_and_no_positive(self):
        ds = SimpleNamespace(_panorama_metadata=pd.DataFrame({
            "pano_id": ["a", "b", "c"],
            "positive_satellite_idxs": [[1], [], [0, 1]],
        }))
        pairs = train_ct2l.build_pairs(ds, {"a": "desc a", "b": "desc b"}, ["/t0.png", "/t1.png"])
        self.assertEqual(pairs, [("desc a", "/t1.png")])


if __name__ == "__main__":
    unittest.main()
