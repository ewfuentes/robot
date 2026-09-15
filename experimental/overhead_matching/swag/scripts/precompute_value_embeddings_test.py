import json
import pickle
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from experimental.overhead_matching.swag.scripts import precompute_value_embeddings


class PrecomputeValueEmbeddingsTest(unittest.TestCase):
    @staticmethod
    def _write_pano_v2_artifact(root: Path, pano_id: str, value: str):
        embeddings_dir = root / "embeddings"
        embeddings_dir.mkdir(parents=True)
        with open(embeddings_dir / "embeddings.pkl", "wb") as output:
            pickle.dump({
                "version": "2.0",
                "panoramas": {
                    f"{pano_id},42.0,-71.0,": {
                        "landmarks": [{
                            "primary_tag": {"key": "name", "value": value},
                            "additional_tags": [],
                            "bounding_boxes": [],
                        }],
                    },
                },
            }, output)

    def test_multiple_direct_pano_v2_artifact_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "charles" / "v1"
            second = Path(temporary) / "washington" / "v1"
            self._write_pano_v2_artifact(first, "pano-a", "Charles River")
            self._write_pano_v2_artifact(second, "pano-b", "Mount Washington")

            self.assertEqual(
                precompute_value_embeddings.collect_text_values_from_pano_v2(
                    [first, second]),
                Counter({"Charles River": 1, "Mount Washington": 1}),
            )

    def test_repeated_pano_ids_across_artifact_roots_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            self._write_pano_v2_artifact(first, "duplicate", "First")
            self._write_pano_v2_artifact(second, "duplicate", "Second")

            self.assertEqual(
                precompute_value_embeddings.collect_text_values_from_pano_v2(
                    [first, second]),
                Counter({"First": 1, "Second": 1}),
            )

    def test_direct_compact_feather_and_artifact_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_dir = Path(temporary) / "osm_artifact"
            artifact_dir.mkdir()
            feather_path = artifact_dir / "landmarks.feather"
            pd.DataFrame({
                "id": ["node:1", "way:2"],
                "geometry": ["POINT (0 0)", "POINT (1 1)"],
                "landmark_type": ["osm", "osm"],
                "tags": [
                    json.dumps({"amenity": "school", "source": "survey"},
                               sort_keys=True, separators=(",", ":")),
                    json.dumps({"name": "Summit Lodge", "tourism": "hotel"},
                               sort_keys=True, separators=(",", ":")),
                ],
            }).to_feather(feather_path)

            expected = Counter({
                "school": 1,
                "Summit Lodge": 1,
                "hotel": 1,
            })
            collect = (
                precompute_value_embeddings.collect_text_values_from_feather)
            for source in (feather_path, artifact_dir):
                with self.subTest(source=source):
                    self.assertEqual(collect([source]), expected)

    def test_legacy_wide_vigor_layout_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            city_dir = Path(temporary) / "LegacyCity"
            landmarks_dir = city_dir / "landmarks"
            landmarks_dir.mkdir(parents=True)
            pd.DataFrame({
                "amenity": ["school", None],
                "name": [None, "Old Hall"],
                "source": ["survey", "survey"],
            }).to_feather(landmarks_dir / "v1.feather")

            self.assertEqual(
                precompute_value_embeddings.collect_text_values_from_feather(
                    [city_dir]),
                Counter({"school": 1, "Old Hall": 1}),
            )


if __name__ == "__main__":
    unittest.main()
