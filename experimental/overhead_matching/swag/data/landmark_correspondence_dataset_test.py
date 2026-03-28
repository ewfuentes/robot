"""Tests for landmark correspondence dataset parsing and collation."""

import json
import math
import tempfile
import unittest
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


def _make_fake_text_embeddings(*values, dim=32):
    """Create a fake text_embeddings dict mapping each value to a random tensor."""
    return {v: torch.randn(dim) for v in values}


class TestTagStringParsing(unittest.TestCase):
    def test_simple(self):
        tags = parse_tag_string("building=yes; name=Library")
        self.assertEqual(tags, {"building": "yes", "name": "Library"})

    def test_single_tag(self):
        tags = parse_tag_string("highway=residential")
        self.assertEqual(tags, {"highway": "residential"})

    def test_value_with_equals(self):
        # addr:housenumber=1252-1414 has no extra =, but test robustness
        tags = parse_tag_string("name=Route 66=Historic")
        self.assertEqual(tags, {"name": "Route 66=Historic"})

    def test_duplicate_keys_last_wins(self):
        tags = parse_tag_string("building=yes; building=commercial")
        self.assertEqual(tags, {"building": "commercial"})


class TestPromptParsing(unittest.TestCase):
    SAMPLE_PROMPT = """Set 1 (street-level observations):
 0. man_made=bridge; bridge=yes
 1. highway=sign; name=Exit 27C

Set 2 (map database):
 0. amenity=parking; access=destination
 1. building=yes; building:levels=3"""

    def test_parse_sets(self):
        set1, set2 = parse_prompt_landmarks(self.SAMPLE_PROMPT)
        self.assertEqual(len(set1), 2)
        self.assertEqual(len(set2), 2)
        self.assertEqual(set1[0], {"man_made": "bridge", "bridge": "yes"})
        self.assertEqual(set2[1], {"building": "yes", "building:levels": "3"})

    def test_landmark_count(self):
        set1, set2 = parse_prompt_landmarks(self.SAMPLE_PROMPT)
        self.assertEqual(len(set1), 2)
        self.assertEqual(len(set2), 2)


class TestJSONLParsing(unittest.TestCase):
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
        self.assertEqual(len(pairs), 3)  # 1 positive + 1 hard + 1 easy

    def test_pair_labels(self):
        entry = self._make_jsonl_entry()
        pairs = parse_jsonl_line(entry)
        labels = {p.difficulty: p.label for p in pairs}
        self.assertEqual(labels["positive"], 1.0)
        self.assertEqual(labels["hard"], 0.0)
        self.assertEqual(labels["easy"], 0.0)

    def test_pair_tags(self):
        entry = self._make_jsonl_entry()
        pairs = parse_jsonl_line(entry)
        pos = [p for p in pairs if p.difficulty == "positive"][0]
        self.assertEqual(pos.pano_tags["name"], "Library")
        self.assertEqual(pos.osm_tags["name"], "Library")

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
        self.assertEqual(len(pos_pairs), 0)

    def test_load_from_directory(self):
        entry = self._make_jsonl_entry()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure matching real data
            resp_dir = Path(tmpdir) / "responses" / "prediction-model-test"
            resp_dir.mkdir(parents=True)
            jsonl_path = resp_dir / "predictions.jsonl"
            jsonl_path.write_text(json.dumps(entry) + "\n")

            pairs = load_pairs_from_directory(Path(tmpdir))
            self.assertEqual(len(pairs), 3)


class TestCrossFeatures(unittest.TestCase):
    # Feature layout (13 total):
    #  [0]     Jaccard similarity
    #  [1]     shared keys / 10
    #  [2]     exact matches / 10
    #  [3..5]  text sim: max, mean, name
    #  [6..11] numeric proximity in sorted key order:
    #          building:levels, heritage, lanes, levels, maxheight, maxspeed
    #  [12]   housenumber overlap

    def test_feature_count(self):
        pano = {"building": "yes", "name": "Library"}
        osm = {"building": "yes", "name": "Library"}
        feats = compute_cross_features(pano, osm)
        self.assertEqual(len(feats), 13)

    def test_numeric_feature_order(self):
        """Numeric proximity features must follow sorted(NUMERIC_KEYS) order."""
        # building:levels is at index 6, lanes at 8, maxspeed at 11
        pano = {"building:levels": "3", "lanes": "2", "maxspeed": "30 mph"}
        osm = {"building:levels": "3", "lanes": "2", "maxspeed": "30 mph"}
        feats = compute_cross_features(pano, osm)
        # Exact matches → exp(0) = 1.0
        self.assertEqual(feats[6], 1.0, "building:levels should be at index 6")
        self.assertEqual(feats[8], 1.0, "lanes should be at index 8")
        self.assertEqual(feats[11], 1.0, "maxspeed should be at index 11")
        # heritage (7), levels (9), maxheight (10) are absent → 0.0
        self.assertEqual(feats[7], 0.0, "heritage absent")
        self.assertEqual(feats[9], 0.0, "levels absent")
        self.assertEqual(feats[10], 0.0, "maxheight absent")

    def test_identical_tags_high_similarity(self):
        tags = {"building": "yes", "name": "Library", "amenity": "library"}
        feats = compute_cross_features(tags, tags)
        # Jaccard should be 1.0
        self.assertEqual(feats[0], 1.0)
        # All exact matches
        self.assertGreater(feats[2], 0)

    def test_disjoint_tags_low_similarity(self):
        pano = {"building": "yes"}
        osm = {"highway": "residential"}
        feats = compute_cross_features(pano, osm)
        # Jaccard should be 0.0
        self.assertEqual(feats[0], 0.0)

    def test_housenumber_range_overlap(self):
        pano = {"addr:housenumber": "665"}
        osm = {"addr:housenumber": "660-670"}
        feats = compute_cross_features(pano, osm)
        # Last feature is housenumber overlap
        self.assertEqual(feats[-1], 1.0)

    def test_housenumber_no_overlap(self):
        pano = {"addr:housenumber": "100"}
        osm = {"addr:housenumber": "660-670"}
        feats = compute_cross_features(pano, osm)
        self.assertEqual(feats[-1], 0.0)

    def test_text_cosine_similarity_features(self):
        """Text sim features (indices 3-5) should be non-zero when embeddings provided."""
        lib_emb = torch.tensor([1.0, 0.0, 0.0])
        library_emb = torch.tensor([0.9, 0.1, 0.0])
        text_embs = {"Library": library_emb, "Lib": lib_emb, "yes": torch.randn(3)}
        pano = {"name": "Lib", "building": "yes"}
        osm = {"name": "Library", "building": "yes"}
        feats = compute_cross_features(pano, osm, text_embeddings=text_embs)
        expected_name_sim = torch.nn.functional.cosine_similarity(
            lib_emb.unsqueeze(0), library_emb.unsqueeze(0)
        ).item()
        expected_building_sim = 1.0  # same value "yes"
        # max text sim
        self.assertAlmostEqual(feats[3], max(expected_name_sim, expected_building_sim), places=4)
        # mean text sim
        self.assertAlmostEqual(feats[4], (expected_name_sim + expected_building_sim) / 2, places=4)
        # name-specific sim
        self.assertAlmostEqual(feats[5], expected_name_sim, places=4)

    def test_text_sim_zero_without_name_key(self):
        """Name-specific sim (index 5) should be 0 when 'name' key not shared."""
        text_embs = {"yes": torch.randn(3), "commercial": torch.randn(3)}
        pano = {"building": "yes"}
        osm = {"building": "commercial"}
        feats = compute_cross_features(pano, osm, text_embeddings=text_embs)
        self.assertEqual(feats[5], 0.0)


class TestTagBundleEncoding(unittest.TestCase):
    def test_text_tag_encoding(self):
        tags = {"name": "Library", "building": "yes"}
        text_embs = _make_fake_text_embeddings("Library", "yes")
        encoded = encode_tag_bundle(tags, text_embs, text_input_dim=32)
        self.assertEqual(len(encoded["key_indices"]), 2)
        self.assertTrue(all(vt == 3 for vt in encoded["value_types"]))  # Both are text

    def test_text_missing_embedding_raises(self):
        tags = {"name": "Library"}
        text_embs = _make_fake_text_embeddings()  # empty
        with self.assertRaises(KeyError):
            encode_tag_bundle(tags, text_embs)

    def test_text_none_embeddings_raises(self):
        tags = {"name": "Library"}
        with self.assertRaises(ValueError):
            encode_tag_bundle(tags, None)

    def test_boolean_tag_encoding(self):
        tags = {"covered": "yes"}
        encoded = encode_tag_bundle(tags, None)
        self.assertEqual(encoded["value_types"][0], 0)  # boolean type
        self.assertEqual(encoded["boolean_values"][0], 0)  # yes → 0 (true)

    def test_numeric_tag_encoding(self):
        tags = {"building:levels": "5"}
        encoded = encode_tag_bundle(tags, None)
        self.assertEqual(encoded["value_types"][0], 1)  # numeric type
        self.assertFalse(encoded["numeric_nan_mask"][0])

    def test_housenumber_encoding(self):
        tags = {"addr:housenumber": "665-667"}
        encoded = encode_tag_bundle(tags, None)
        self.assertEqual(encoded["value_types"][0], 2)  # housenumber type
        self.assertFalse(encoded["housenumber_nan_mask"][0])

    def test_unknown_key_skipped(self):
        tags = {"unknown_key_xyz": "value"}
        encoded = encode_tag_bundle(tags, None)
        self.assertEqual(len(encoded["key_indices"]), 0)


class TestCollation(unittest.TestCase):
    def test_collate_batch(self):
        text_embs = _make_fake_text_embeddings(
            "yes", "Lib", "Library", "residential", "restaurant", "Portillos",
        )
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
            pairs, text_embeddings=text_embs, text_input_dim=32,
            include_difficulties=("positive", "easy"),
        )
        samples = [dataset[0], dataset[1]]
        batch = collate_correspondence(samples)

        self.assertEqual(batch.pano_key_indices.shape[0], 2)  # batch size
        self.assertEqual(batch.osm_key_indices.shape[0], 2)
        self.assertEqual(batch.labels.shape, (2,))
        self.assertEqual(batch.cross_features.shape, (2, 13))
        self.assertEqual(batch.labels[0], 1.0)
        self.assertEqual(batch.labels[1], 0.0)

    def test_collate_with_unknown_keys(self):
        """Samples where all tag keys are unrecognized should not crash collation."""
        text_embs = _make_fake_text_embeddings("yes")
        pairs = [
            CorrespondencePair(
                pano_tags={"unknown_xyz": "val"},
                osm_tags={"building": "yes"},
                label=0.0, difficulty="easy", uniqueness_score=1, pano_id="p1",
            ),
            CorrespondencePair(
                pano_tags={"building": "yes"},
                osm_tags={"building": "yes"},
                label=1.0, difficulty="positive", uniqueness_score=3, pano_id="p2",
            ),
        ]
        dataset = LandmarkCorrespondenceDataset(
            pairs, text_embeddings=text_embs, text_input_dim=32,
            include_difficulties=("positive", "easy"),
        )
        batch = collate_correspondence([dataset[0], dataset[1]])
        self.assertEqual(batch.pano_key_indices.shape[0], 2)
        # First sample's pano side has no recognized keys → mask should be all False
        self.assertFalse(batch.pano_tag_mask[0].any())

    def test_collate_to_device(self):
        text_embs = _make_fake_text_embeddings("yes")
        pairs = [
            CorrespondencePair(
                pano_tags={"building": "yes"},
                osm_tags={"building": "yes"},
                label=1.0, difficulty="positive", uniqueness_score=3, pano_id="p1",
            ),
        ]
        dataset = LandmarkCorrespondenceDataset(
            pairs, text_embeddings=text_embs, text_input_dim=32,
            include_difficulties=("positive",),
        )
        batch = collate_correspondence([dataset[0]])
        batch_cpu = batch.to(torch.device("cpu"))
        self.assertEqual(batch_cpu.labels.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
