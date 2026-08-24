import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.dataset_tools import (
    dataset_status_table as subject,
)


def dataset(root: Path) -> Path:
    root.mkdir()
    (root / "pipeline_metadata.json").write_text(json.dumps({
        "projection": "equirectangular", "num_images": 0,
    }))
    (root / "frames_gps.csv").write_text("idx,dist_m\n")
    return root


class CorruptJsonTest(unittest.TestCase):

    def test_corrupt_sidecars_are_distinct_from_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = dataset(root / "clean")
            corrupt = dataset(root / "corrupt")
            manifests = corrupt / "_manifests"
            manifests.mkdir()
            (manifests / "vehicle_anchor.json").write_text("{broken")
            (manifests / "recording_seams.json").write_text("[broken")
            clean_row = subject.collect_one(clean)
            corrupt_row = subject.collect_one(corrupt)
            self.assertEqual(clean_row["anchor"], "—")
            self.assertIsNone(clean_row["seams"])
            self.assertEqual(corrupt_row["anchor"], subject.CORRUPT)
            self.assertEqual(corrupt_row["seams"], subject.CORRUPT)
            self.assertIn(subject.CORRUPT,
                          subject.render([clean_row, corrupt_row]))

    def test_corrupt_pipeline_metadata_still_gets_a_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            (root / "pipeline_metadata.json").write_text("not-json")
            row = subject.collect_one(root)
            self.assertIsNotNone(row)
            self.assertEqual(row["proj"], subject.CORRUPT)

    def test_valid_json_with_the_wrong_shape_is_corrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = dataset(Path(directory) / "dataset")
            (root / "pipeline_metadata.json").write_text("[]")
            row = subject.collect_one(root)
            self.assertIsNotNone(row)
            self.assertEqual(row["proj"], subject.CORRUPT)


if __name__ == "__main__":
    unittest.main()
