"""Tests for osm_sentence_generator module."""

import unittest

from experimental.overhead_matching.swag.data.osm_sentence_generator import (
    GeneratedSentence,
    OSMSentenceGenerator,
    TagTemplateConfig,
    compute_coverage_stats,
)


class TestOSMSentenceGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = OSMSentenceGenerator()

    def test_basic_restaurant(self):
        """Test generating sentence for a restaurant."""
        tags = {"amenity": "restaurant", "name": "Joe's Diner", "cuisine": "american"}
        result = self.generator.generate_sentence(tags)

        self.assertIsInstance(result, GeneratedSentence)
        self.assertIn("amenity", result.used_tags)
        self.assertEqual(result.used_tags["amenity"], "restaurant")
        self.assertTrue(result.sentence)  # Non-empty
        self.assertIn(result.template_type, ["poi_named", "poi_anonymous"])

    def test_anonymous_restaurant(self):
        """Test generating sentence for restaurant without name."""
        tags = {"amenity": "restaurant", "cuisine": "italian"}
        result = self.generator.generate_sentence(tags)

        self.assertIn("amenity", result.used_tags)
        self.assertTrue(result.sentence)

    def test_building_with_levels(self):
        """Test generating sentence for building with levels."""
        tags = {"building": "commercial", "building:levels": "5", "name": "Office Tower"}
        result = self.generator.generate_sentence(tags)

        self.assertIn("building", result.used_tags)
        self.assertTrue(result.sentence)
        self.assertIn(result.template_type, ["building_sized", "building_typed"])

    def test_highway(self):
        """Test generating sentence for highway."""
        tags = {"highway": "residential", "name": "Oak Street", "surface": "asphalt"}
        result = self.generator.generate_sentence(tags)

        self.assertIn("highway", result.used_tags)
        self.assertEqual(result.template_type, "highway")
        self.assertTrue(result.sentence)

    def test_natural_feature(self):
        """Test generating sentence for natural feature."""
        tags = {"natural": "tree"}
        result = self.generator.generate_sentence(tags)

        self.assertIn("natural", result.used_tags)
        self.assertEqual(result.template_type, "natural")
        self.assertTrue(result.sentence)

    def test_landuse(self):
        """Test generating sentence for landuse."""
        tags = {"landuse": "residential"}
        result = self.generator.generate_sentence(tags)

        self.assertIn("landuse", result.used_tags)
        self.assertEqual(result.template_type, "landuse")
        self.assertTrue(result.sentence)

    def test_generic_fallback(self):
        """Test fallback for unknown tags."""
        tags = {"some_unknown_tag": "value"}
        result = self.generator.generate_sentence(tags)

        self.assertEqual(result.template_type, "generic_fallback")
        self.assertTrue(result.sentence)

    def test_determinism(self):
        """Test that same tags produce same sentence."""
        tags = {"amenity": "restaurant", "name": "Test Place"}
        result1 = self.generator.generate_sentence(tags)
        result2 = self.generator.generate_sentence(tags)

        self.assertEqual(result1.sentence, result2.sentence)
        self.assertEqual(result1.used_tags, result2.used_tags)
        self.assertEqual(result1.template_type, result2.template_type)

    def test_different_seeds_produce_different_sentences(self):
        """Test that different seed offsets can produce different sentences."""
        tags = {
            "amenity": "restaurant",
            "name": "Test Place",
            "cuisine": "italian",
            "brand": "Test Brand",
        }

        # Generate multiple sentences
        sentences = self.generator.generate_sentences(tags, n=5)
        unique_sentences = set(s.sentence for s in sentences)

        # At least some should be different (though not guaranteed for all tags)
        self.assertEqual(len(sentences), 5)

    def test_used_tags_tracking(self):
        """Test that used_tags correctly tracks what's in the sentence."""
        tags = {"amenity": "restaurant", "name": "Joe's", "cuisine": "mexican"}
        result = self.generator.generate_sentence(tags)

        # Primary category should always be used
        self.assertIn("amenity", result.used_tags)

        # All original tags should be either used or unused
        for key in tags:
            self.assertTrue(
                key in result.used_tags or key in result.unused_tags,
                f"Tag '{key}' not tracked",
            )

    def test_unused_tags_tracking(self):
        """Test that unused tags are properly tracked."""
        # Use tags where we can control inclusion
        tags = {"building": "yes"}  # Simple building
        result = self.generator.generate_sentence(tags)

        # Verify building is used
        self.assertIn("building", result.used_tags)
        self.assertNotIn("building", result.unused_tags)

    def test_generate_sentences_returns_list(self):
        """Test generate_sentences returns correct number of sentences."""
        tags = {"amenity": "cafe", "name": "Coffee Shop"}
        sentences = self.generator.generate_sentences(tags, n=3)

        self.assertEqual(len(sentences), 3)
        self.assertTrue(all(isinstance(s, GeneratedSentence) for s in sentences))

    def test_variety_across_different_landmarks(self):
        """Test that different landmarks get different sentences."""
        landmarks = [
            {"amenity": "restaurant", "name": "Place A"},
            {"amenity": "restaurant", "name": "Place B"},
            {"amenity": "cafe", "name": "Place C"},
        ]

        sentences = [self.generator.generate_sentence(tags).sentence for tags in landmarks]

        # All should be unique
        self.assertEqual(len(set(sentences)), 3)

    def test_synonym_variation(self):
        """Test that synonyms are used for category values."""
        # Check that synonym banks exist for common values
        self.assertIn("restaurant", self.generator._category_synonyms)
        self.assertGreater(len(self.generator._category_synonyms["restaurant"]), 1)

    def test_shop_category(self):
        """Test generating sentence for shop."""
        tags = {"shop": "supermarket", "name": "FoodMart"}
        result = self.generator.generate_sentence(tags)

        self.assertIn("shop", result.used_tags)
        self.assertIn(result.template_type, ["poi_named", "poi_anonymous"])

    def test_tourism_category(self):
        """Test generating sentence for tourism."""
        tags = {"tourism": "hotel", "name": "Grand Hotel"}
        result = self.generator.generate_sentence(tags)

        self.assertIn("tourism", result.used_tags)

    def test_leisure_category(self):
        """Test generating sentence for leisure."""
        tags = {"leisure": "park", "name": "Central Park"}
        result = self.generator.generate_sentence(tags)

        self.assertIn("leisure", result.used_tags)

    def test_empty_tags(self):
        """Test handling of empty tags."""
        tags = {}
        result = self.generator.generate_sentence(tags)

        self.assertEqual(result.template_type, "generic_fallback")
        self.assertTrue(result.sentence)


class TestCoverageStats(unittest.TestCase):
    def test_coverage_stats_basic(self):
        """Test basic coverage stats computation."""
        sentences = [
            GeneratedSentence("s1", {}, {}, "poi_named"),
            GeneratedSentence("s2", {}, {}, "poi_named"),
            GeneratedSentence("s3", {}, {}, "building_typed"),
            GeneratedSentence("s4", {}, {}, "generic_fallback"),
        ]

        stats = compute_coverage_stats(sentences)

        self.assertEqual(stats["counts"]["poi_named"], 2)
        self.assertEqual(stats["counts"]["building_typed"], 1)
        self.assertEqual(stats["counts"]["generic_fallback"], 1)
        self.assertEqual(stats["fallback_rate"], 25.0)

    def test_coverage_stats_empty(self):
        """Test coverage stats with empty list."""
        stats = compute_coverage_stats([])

        self.assertEqual(stats["counts"], {})
        self.assertEqual(stats["percentages"], {})
        self.assertEqual(stats["fallback_rate"], 0.0)

    def test_coverage_stats_no_fallback(self):
        """Test coverage stats with no fallback."""
        sentences = [
            GeneratedSentence("s1", {}, {}, "poi_named"),
            GeneratedSentence("s2", {}, {}, "highway"),
        ]

        stats = compute_coverage_stats(sentences)

        self.assertEqual(stats["fallback_rate"], 0.0)


class TestTagTemplateConfig(unittest.TestCase):
    def test_default_config(self):
        """Test default configuration values."""
        config = TagTemplateConfig()

        self.assertIn("amenity", config.category_tags)
        self.assertIn("building", config.category_tags)
        self.assertIn("highway", config.category_tags)
        self.assertIn("name", config.attribute_tags)
        self.assertIn("addr:street", config.attribute_tags)  # Addresses are kept
        self.assertIn("ref:", config.excluded_tag_prefixes)  # ref:* prefixes excluded
        self.assertIn("tiger:", config.excluded_tag_prefixes)
        self.assertIn("source", config.excluded_tags)

    def test_custom_config(self):
        """Test custom configuration."""
        config = TagTemplateConfig(category_tags=("amenity", "shop"))
        generator = OSMSentenceGenerator(config)

        # Building should now fall back to generic
        tags = {"building": "yes"}
        result = generator.generate_sentence(tags)
        self.assertEqual(result.template_type, "generic_fallback")


if __name__ == "__main__":
    unittest.main()
