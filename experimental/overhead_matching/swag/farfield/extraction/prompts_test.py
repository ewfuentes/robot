import hashlib
import json
import unittest

from experimental.overhead_matching.swag.farfield.extraction import prompts

# Pinned digests of the prompt TEXT. These are the digests recorded in every
# frame_landmarks manifest built with these prompts; if one of these tests
# fails, the prompt text changed, which invalidates comparisons against every
# extraction that recorded the old digest. Change the text only under a NEW
# registry key.
PROMPT_SHA256 = {
    "osm_tags_farfield":
        "57be8dabaf91dbf9514ab881d349b5618d8dc40b34ad835f0f3afa7e76b609ac",
    "osm_tags_farfield_v2":
        "4a914d73cc4a3f300f7e7ee26761d7b18932ab17ca14dace85da9bffe1026cbe",
}
USER_PROMPT_SHA256 = (
    "c190ff660bff6c29706f87ac92bf0431065518853ff881b32f35e06e18d435e5")


class PromptRegistryTest(unittest.TestCase):
    def test_registry_contains_exactly_the_farfield_prompts(self):
        self.assertEqual(set(prompts.SYSTEM_PROMPTS), set(PROMPT_SHA256))
        self.assertEqual(prompts.PROMPT_TYPES,
                         tuple(sorted(PROMPT_SHA256)))

    def test_prompt_text_digests_are_stable(self):
        for name, want in PROMPT_SHA256.items():
            got = hashlib.sha256(
                prompts.SYSTEM_PROMPTS[name].encode()).hexdigest()
            self.assertEqual(got, want, f"{name} text changed")
            self.assertEqual(prompts.prompt_sha256(name), want)

    def test_user_prompt_digest_is_stable(self):
        self.assertEqual(
            hashlib.sha256(prompts.USER_PROMPT.encode()).hexdigest(),
            USER_PROMPT_SHA256)

    def test_v2_carries_the_structure_not_scene_clause(self):
        # The clause the v1->v2 revision exists for: a name must come from the
        # structure's own visible features, never from the scene resembling a
        # known place (the Chicago-lakefront failure mode).
        v2 = prompts.SYSTEM_PROMPTS["osm_tags_farfield_v2"]
        self.assertIn("rest on the overall view resembling a place you know",
                      v2)
        v1 = prompts.SYSTEM_PROMPTS["osm_tags_farfield"]
        self.assertNotIn("rest on the overall view", v1)
        self.assertIn("Similar-looking neighbours are the normal case", v1)

    def test_the_prompt_text_has_exactly_one_owner(self):
        """The registries must BE the owner's strings, not copies of them.

        This is the property the duplication removal bought: an extraction
        run records prompt_sha256, so two registries holding their own copy
        means two runs can claim the same prompt_type while having been
        given different instructions. Identity (`is`) rather than equality,
        because equality is exactly what silently drifts.
        """
        from experimental.overhead_matching.swag.model import (
            farfield_prompt_text,
        )
        for name, text in prompts.SYSTEM_PROMPTS.items():
            self.assertIs(text,
                          farfield_prompt_text.FARFIELD_SYSTEM_PROMPTS[name],
                          name)
        self.assertIs(prompts.USER_PROMPT,
                      farfield_prompt_text.OSM_TAGS_USER_PROMPT)

    def test_the_shared_extractor_reads_the_same_owner(self):
        """The other consumer, checked by digest rather than by import: the
        shared extractor pulls torch, so this test reads its source."""
        import ast
        import hashlib
        from pathlib import Path
        from experimental.overhead_matching.swag.model import (
            farfield_prompt_text,
        )
        source = Path(farfield_prompt_text.__file__).with_name(
            "semantic_landmark_extractor.py").read_text()
        tree = ast.parse(source)
        registry = next(
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "SYSTEM_PROMPTS"
                    for t in node.targets))
        farfield_entries = {
            key.value: value for key, value in zip(registry.keys,
                                                   registry.values)
            if isinstance(key, ast.Constant)
            and key.value.startswith("osm_tags_farfield")}
        self.assertEqual(set(farfield_entries), set(prompts.SYSTEM_PROMPTS))
        for name, node in farfield_entries.items():
            # An Attribute node is a reference to the owner; a Constant would
            # mean the text had been pasted back in.
            self.assertIsInstance(
                node, ast.Attribute,
                f"{name} is inlined in semantic_landmark_extractor: the "
                f"prompt text must reference farfield_prompt_text")
            digest = hashlib.sha256(
                prompts.SYSTEM_PROMPTS[name].encode()).hexdigest()
            self.assertEqual(prompts.prompt_sha256(name), digest)

    def test_unknown_prompt_type_raises(self):
        with self.assertRaises(KeyError):
            prompts.prompt_sha256("osm_tags")


class ResponseSchemaTest(unittest.TestCase):
    def test_schema_is_fully_inlined(self):
        schema = prompts.response_schema()
        text = json.dumps(schema)
        self.assertNotIn("$ref", text)
        self.assertNotIn("$defs", text)
        self.assertNotIn('"title"', text)

    def test_schema_shape_matches_what_ingest_reads(self):
        schema = prompts.response_schema()
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["location_type", "landmarks"])
        landmark = schema["properties"]["landmarks"]["items"]
        self.assertEqual(
            set(landmark["properties"]),
            {"primary_tag", "additional_tags", "confidence",
             "bounding_boxes", "description"})
        box = landmark["properties"]["bounding_boxes"]["items"]
        self.assertEqual(set(box["properties"]),
                         {"yaw_angle", "ymin", "xmin", "ymax", "xmax"})
        self.assertEqual(box["properties"]["xmax"]["maximum"], 1000)

    def test_primary_tag_enum_includes_place(self):
        # The one deliberate difference from main's OSM-tag schema: the
        # farfield prompts direct islands/settlements to place=*.
        schema = prompts.response_schema()
        landmark = schema["properties"]["landmarks"]["items"]
        enum = landmark["properties"]["primary_tag"]["properties"]["key"][
            "enum"]
        self.assertIn("place", enum)
        self.assertIn("man_made", enum)
        self.assertIn("seamark:type", json.dumps(prompts.SYSTEM_PROMPTS[
            "osm_tags_farfield"]))  # prompt and schema evolve together


class BuildRequestTest(unittest.TestCase):
    IMAGES = [("image/jpeg", "AAAA"), ("image/jpeg", "BBBB"),
              ("image/jpeg", "CCCC"), ("image/jpeg", "DDDD")]

    def test_high_resolution_is_set_globally(self):
        record = prompts.build_request(
            "stem0", self.IMAGES, prompt_type="osm_tags_farfield_v2",
            media_resolution="MEDIA_RESOLUTION_HIGH", thinking_level="HIGH")
        self.assertEqual(record["key"], "stem0")
        request = record["request"]
        parts = request["contents"][0]["parts"]
        self.assertEqual(len(parts), 5)  # 4 images + user prompt
        for part in parts[:4]:
            self.assertIn("inline_data", part)
            self.assertNotIn("media_resolution", part)
        self.assertEqual(parts[4]["text"], prompts.USER_PROMPT)
        config = request["generationConfig"]
        self.assertEqual(config["mediaResolution"], "MEDIA_RESOLUTION_HIGH")
        self.assertEqual(config["thinkingConfig"]["thinkingLevel"], "HIGH")
        self.assertEqual(config["responseMimeType"], "application/json")
        self.assertEqual(config["responseSchema"], prompts.response_schema())
        self.assertEqual(
            request["systemInstruction"]["parts"][0]["text"],
            prompts.SYSTEM_PROMPTS["osm_tags_farfield_v2"])

    def test_ultra_high_is_set_per_part(self):
        # A Gemini API quirk preserved from the reference implementation:
        # ULTRA_HIGH goes on each image part, not in generationConfig.
        record = prompts.build_request(
            "stem0", self.IMAGES, prompt_type="osm_tags_farfield",
            media_resolution="MEDIA_RESOLUTION_ULTRA_HIGH",
            thinking_level="LOW")
        request = record["request"]
        parts = request["contents"][0]["parts"]
        for part in parts[:4]:
            self.assertEqual(part["media_resolution"],
                             {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"})
        self.assertNotIn("mediaResolution", request["generationConfig"])

    def test_unknown_prompt_type_raises(self):
        with self.assertRaises(KeyError):
            prompts.build_request(
                "stem0", self.IMAGES, prompt_type="panorama",
                media_resolution="MEDIA_RESOLUTION_HIGH",
                thinking_level="HIGH")


if __name__ == "__main__":
    unittest.main()
