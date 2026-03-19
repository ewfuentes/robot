"""Tests for landmark correspondence dataset parsing and collation."""

import json
import math
import tempfile
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    CorrespondencePair,
    LandmarkCorrespondenceDataset,
    collate_correspondence,
    compute_cross_features,
    encode_tag_bundle,
    load_pairs_from_directory,
    parse_jsonl_line,
    parse_prompt_landmarks,
    parse_tag_string,
)


class TestTagStringParsing:
    def test_simple(self):
        tags = parse_tag_string("building=yes; name=Library")
        assert tags == {"building": "yes", "name": "Library"}

    def test_single_tag(self):
        tags = parse_tag_string("highway=residential")
        assert tags == {"highway": "residential"}

    def test_value_with_equals(self):
        # addr:housenumber=1252-1414 has no extra =, but test robustness
        tags = parse_tag_string("name=Route 66=Historic")
        assert tags == {"name": "Route 66=Historic"}


class TestPromptParsing:
    SAMPLE_PROMPT = """Set 1 (street-level observations):
 0. man_made=bridge; bridge=yes
 1. highway=sign; name=Exit 27C

Set 2 (map database):
 0. amenity=parking; access=destination
 1. building=yes; building:levels=3"""

    def test_parse_sets(self):
        set1, set2 = parse_prompt_landmarks(self.SAMPLE_PROMPT)
        assert len(set1) == 2
        assert len(set2) == 2
        assert set1[0] == {"man_made": "bridge", "bridge": "yes"}
        assert set2[1] == {"building": "yes", "building:levels": "3"}

    def test_landmark_count(self):
        set1, set2 = parse_prompt_landmarks(self.SAMPLE_PROMPT)
        assert len(set1) == 2
        assert len(set2) == 2


class TestJSONLParsing:
    def _make_jsonl_entry(self):
        return {
            "key": "test_pano_id",
            "request": {
                "contents": [{
                    "parts": [{"text": """Set 1 (street-level observations):
 0. building=yes; name=Library
 1. highway=residential

Set 2 (map database):
 0. building=commercial; name=Library
 1. amenity=restaurant; name=Portillos
 2. highway=residential; name=Main St"""}],
                    "role": "user",
                }]
            },
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{"text": json.dumps({
                            "matches": [{
                                "set_1_id": 0,
                                "set_2_matches": [0],
                                "uniqueness_score": 4,
                                "negatives": [
                                    {"set_2_id": 1, "difficulty": "hard"},
                                    {"set_2_id": 2, "difficulty": "easy"},
                                ]
                            }]
                        })}]
                    }
                }]
            }
        }

    def test_parse_produces_pairs(self):
        entry = self._make_jsonl_entry()
        pairs = parse_jsonl_line(entry)
        assert len(pairs) == 3  # 1 positive + 1 hard + 1 easy

    def test_pair_labels(self):
        entry = self._make_jsonl_entry()
        pairs = parse_jsonl_line(entry)
        labels = {p.difficulty: p.label for p in pairs}
        assert labels["positive"] == 1.0
        assert labels["hard"] == 0.0
        assert labels["easy"] == 0.0

    def test_pair_tags(self):
        entry = self._make_jsonl_entry()
        pairs = parse_jsonl_line(entry)
        pos = [p for p in pairs if p.difficulty == "positive"][0]
        assert pos.pano_tags["name"] == "Library"
        assert pos.osm_tags["name"] == "Library"

    def test_out_of_bounds_ids_skipped(self):
        entry = self._make_jsonl_entry()
        # Set an out-of-bounds match
        resp = json.loads(
            entry["response"]["candidates"][0]["content"]["parts"][0]["text"]
        )
        resp["matches"][0]["set_2_matches"] = [99]
        entry["response"]["candidates"][0]["content"]["parts"][0]["text"] = json.dumps(resp)
        pairs = parse_jsonl_line(entry)
        # Should skip the out-of-bounds positive, keep negatives
        pos_pairs = [p for p in pairs if p.difficulty == "positive"]
        assert len(pos_pairs) == 0

    def test_load_from_directory(self):
        entry = self._make_jsonl_entry()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure matching real data
            resp_dir = Path(tmpdir) / "responses" / "prediction-model-test"
            resp_dir.mkdir(parents=True)
            jsonl_path = resp_dir / "predictions.jsonl"
            jsonl_path.write_text(json.dumps(entry) + "\n")

            pairs = load_pairs_from_directory(Path(tmpdir))
            assert len(pairs) == 3


class TestCrossFeatures:
    def test_feature_count(self):
        pano = {"building": "yes", "name": "Library"}
        osm = {"building": "yes", "name": "Library"}
        feats = compute_cross_features(pano, osm)
        assert len(feats) == 17

    def test_identical_tags_high_similarity(self):
        tags = {"building": "yes", "name": "Library", "amenity": "library"}
        feats = compute_cross_features(tags, tags)
        # Jaccard should be 1.0
        assert feats[0] == 1.0
        # All exact matches
        assert feats[2] > 0

    def test_disjoint_tags_low_similarity(self):
        pano = {"building": "yes"}
        osm = {"highway": "residential"}
        feats = compute_cross_features(pano, osm)
        # Jaccard should be 0.0
        assert feats[0] == 0.0

    def test_housenumber_range_overlap(self):
        pano = {"addr:housenumber": "665"}
        osm = {"addr:housenumber": "660-670"}
        feats = compute_cross_features(pano, osm)
        # Last feature is housenumber overlap
        assert feats[-1] == 1.0

    def test_housenumber_no_overlap(self):
        pano = {"addr:housenumber": "100"}
        osm = {"addr:housenumber": "660-670"}
        feats = compute_cross_features(pano, osm)
        assert feats[-1] == 0.0


class TestTagBundleEncoding:
    def test_text_tag_encoding(self):
        tags = {"name": "Library", "building": "yes"}
        encoded = encode_tag_bundle(tags, None)
        assert len(encoded["key_indices"]) == 2
        assert all(vt == 3 for vt in encoded["value_types"])  # Both are text

    def test_boolean_tag_encoding(self):
        tags = {"covered": "yes"}
        encoded = encode_tag_bundle(tags, None)
        assert encoded["value_types"][0] == 0  # boolean type
        assert encoded["boolean_values"][0] == 0  # yes → 0 (true)

    def test_numeric_tag_encoding(self):
        tags = {"building:levels": "5"}
        encoded = encode_tag_bundle(tags, None)
        assert encoded["value_types"][0] == 1  # numeric type
        assert not encoded["numeric_nan_mask"][0]

    def test_housenumber_encoding(self):
        tags = {"addr:housenumber": "665-667"}
        encoded = encode_tag_bundle(tags, None)
        assert encoded["value_types"][0] == 2  # housenumber type
        assert not encoded["housenumber_nan_mask"][0]

    def test_unknown_key_skipped(self):
        tags = {"unknown_key_xyz": "value"}
        encoded = encode_tag_bundle(tags, None)
        assert len(encoded["key_indices"]) == 0


class TestCollation:
    def test_collate_batch(self):
        pairs = [
            CorrespondencePair(
                pano_tags={"building": "yes", "name": "Lib"},
                osm_tags={"building": "yes", "name": "Library"},
                label=1.0, difficulty="positive", uniqueness_score=4, pano_id="p1",
            ),
            CorrespondencePair(
                pano_tags={"highway": "residential"},
                osm_tags={"amenity": "restaurant", "name": "Portillos"},
                label=0.0, difficulty="easy", uniqueness_score=2, pano_id="p1",
            ),
        ]
        dataset = LandmarkCorrespondenceDataset(
            pairs, text_embeddings=None, include_difficulties=("positive", "easy"),
        )
        samples = [dataset[0], dataset[1]]
        batch = collate_correspondence(samples)

        assert batch.pano_key_indices.shape[0] == 2  # batch size
        assert batch.osm_key_indices.shape[0] == 2
        assert batch.labels.shape == (2,)
        assert batch.cross_features.shape == (2, 17)
        assert batch.labels[0] == 1.0
        assert batch.labels[1] == 0.0

    def test_collate_to_device(self):
        pairs = [
            CorrespondencePair(
                pano_tags={"building": "yes"},
                osm_tags={"building": "yes"},
                label=1.0, difficulty="positive", uniqueness_score=3, pano_id="p1",
            ),
        ]
        dataset = LandmarkCorrespondenceDataset(
            pairs, include_difficulties=("positive",),
        )
        batch = collate_correspondence([dataset[0]])
        batch_cpu = batch.to(torch.device("cpu"))
        assert batch_cpu.labels.device.type == "cpu"
