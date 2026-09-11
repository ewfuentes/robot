import hashlib
import json
import unittest

from experimental.overhead_matching.swag.farfield import dataset
from experimental.overhead_matching.swag.farfield.catalog import catalog
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
    "osm_tags_farfield_v3":
        "e57a97e2c4f27ea9412702f8cfda924fd8892bcad56fdc5b49ed24ea0ea3bad4",
    "osm_tags_farfield_v3_down30":
        "1650d125523a2e97e75cc5ea6552dffe61d89c5efc6fdef6d61e2779a3aaabdb",
}
# response_schema_sha256 recorded in every frame_landmarks manifest built with
# the legacy prompts; the enum those prompts carry is frozen by this digest.
LEGACY_SCHEMA_SHA256 = (
    "05fbc4649e8deb0f90f459066b5062d6e9ddb72b521012403f034cb87a6527f9")


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()
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

    def test_unknown_prompt_type_raises(self):
        with self.assertRaises(KeyError):
            prompts.prompt_sha256("osm_tags")
        with self.assertRaises(KeyError):
            prompts.response_schema("osm_tags")

    def test_every_prompt_declares_its_schema_and_render_pitch(self):
        self.assertEqual(set(prompts.PRIMARY_TAG_KEYS), set(prompts.SYSTEM_PROMPTS))
        self.assertEqual(set(prompts.PROMPT_PITCH_DEG),
                         set(prompts.SYSTEM_PROMPTS))

    def test_v3_generalises_platform_and_measures_ground_range(self):
        v3 = prompts.SYSTEM_PROMPTS["osm_tags_farfield_v3"]
        self.assertIn("or an aircraft", v3)
        self.assertIn("Nothing on it can be a landmark", v3)
        self.assertNotIn("deck, railings", v3)
        self.assertNotIn("Scan the full horizon", v3)
        self.assertNotIn("The setting may be any outdoor environment", v3)
        self.assertIn("measured along the ground from the point directly "
                      "beneath the camera", v3)
        self.assertIn("Land and industry", v3)
        # v2's naming rules travel unchanged.
        self.assertIn("rest on the overall view resembling a place you know",
                      v3)
        self.assertNotIn("__CAMERA_GEOMETRY__", v3)

    def test_v3_down30_differs_from_v3_only_in_camera_geometry(self):
        level = prompts.SYSTEM_PROMPTS["osm_tags_farfield_v3"].splitlines()
        down = prompts.SYSTEM_PROMPTS[
            "osm_tags_farfield_v3_down30"].splitlines()
        self.assertEqual(len(level), len(down))
        changed = [(a, b) for a, b in zip(level, down) if a != b]
        self.assertEqual(len(changed), 1)
        self.assertIn("The camera is level", changed[0][0])
        self.assertIn("pitched 30° below the horizon", changed[0][1])
        self.assertEqual(prompts.PROMPT_PITCH_DEG["osm_tags_farfield_v3"], 0)
        self.assertEqual(
            prompts.PROMPT_PITCH_DEG["osm_tags_farfield_v3_down30"], -30)


class ResponseSchemaTest(unittest.TestCase):
    def test_legacy_schema_digest_matches_recorded_artifacts(self):
        for name in ("osm_tags_farfield", "osm_tags_farfield_v2"):
            self.assertEqual(canonical_sha256(prompts.response_schema(name)),
                             LEGACY_SCHEMA_SHA256, name)

    def test_v3_primary_keys_are_the_catalog_structural_keys(self):
        for name in ("osm_tags_farfield_v3", "osm_tags_farfield_v3_down30"):
            schema = prompts.response_schema(name)
            enum = schema["properties"]["landmarks"]["items"]["properties"][
                "primary_tag"]["properties"]["key"]["enum"]
            self.assertEqual(enum, list(prompts.STRUCTURAL_PRIMARY_TAG_KEYS))
            self.assertEqual(set(enum),
                             catalog.STRUCTURAL_KEYS - {"object_class"})
            for key in ("aeroway", "waterway", "seamark:type", "bridge",
                        "aerialway", "place"):
                self.assertIn(key, enum)
            for key in ("shop", "highway", "office", "public_transport"):
                self.assertNotIn(key, enum)
            self.assertNotEqual(canonical_sha256(schema), LEGACY_SCHEMA_SHA256)
        # Every key the schema admits is named in the prompt's tag guide.
        guide = prompts.SYSTEM_PROMPTS["osm_tags_farfield_v3"]
        for key in prompts.STRUCTURAL_PRIMARY_TAG_KEYS:
            self.assertIn(f"`{key}`", guide, key)

    def test_v3_requires_distance_estimate_as_an_enum_field(self):
        for name in ("osm_tags_farfield_v3", "osm_tags_farfield_v3_down30"):
            landmark = prompts.response_schema(name)["properties"][
                "landmarks"]["items"]
            self.assertIn("distance_estimate", landmark["required"])
            self.assertEqual(
                landmark["properties"]["distance_estimate"]["enum"],
                list(dataset.DISTANCE_BUCKETS))
            self.assertIn("Set distance_estimate to exactly one of",
                          prompts.SYSTEM_PROMPTS[name])
        legacy = prompts.response_schema("osm_tags_farfield")["properties"][
            "landmarks"]["items"]
        self.assertNotIn("distance_estimate", legacy["properties"])
        self.assertNotIn("one of many identical ones",
                         prompts.SYSTEM_PROMPTS["osm_tags_farfield_v3"])

    def test_schema_is_fully_inlined(self):
        schema = prompts.response_schema("osm_tags_farfield")
        text = json.dumps(schema)
        self.assertNotIn("$ref", text)
        self.assertNotIn("$defs", text)
        self.assertNotIn('"title"', text)

    def test_schema_shape_matches_what_ingest_reads(self):
        schema = prompts.response_schema("osm_tags_farfield")
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
        schema = prompts.response_schema("osm_tags_farfield")
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
        self.assertEqual(config["responseSchema"], prompts.response_schema("osm_tags_farfield"))
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

    def test_every_media_resolution_round_trips_to_exact_online_placement(self):
        for resolution in prompts.MEDIA_RESOLUTIONS:
            with self.subTest(resolution=resolution):
                record = prompts.build_request(
                    "stem0", self.IMAGES,
                    prompt_type="osm_tags_farfield_v2",
                    media_resolution=resolution,
                    thinking_level="MEDIUM")
                semantic = prompts.semantic_request_from_batch(
                    record["key"], record["request"])
                self.assertEqual(semantic.media_resolution, resolution)
                self.assertEqual(prompts.batch_record(semantic), record)

                online = prompts.online_request(semantic)
                config = online["config"]
                self.assertEqual(
                    config["thinking_config"], {
                        "thinking_level": "MEDIUM",
                    })
                image_parts = online["contents"][0]["parts"][:4]
                if resolution == "MEDIA_RESOLUTION_ULTRA_HIGH":
                    self.assertNotIn("media_resolution", config)
                    self.assertTrue(all(
                        part["media_resolution"] == {"level": resolution}
                        for part in image_parts))
                else:
                    self.assertEqual(config["media_resolution"], resolution)
                    self.assertTrue(all(
                        "media_resolution" not in part
                        for part in image_parts))

    def test_no_media_resolution_is_valid_for_text_or_audit_requests(self):
        semantic = prompts.semantic_request(
            "audit",
            system_instruction="audit system",
            parts=[
                {"text": "dossier"},
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": "AAAA",
                }},
            ],
            response_schema={"type": "object"},
            thinking_level="HIGH",
        )
        batch = prompts.batch_record(semantic)
        self.assertNotIn(
            "mediaResolution",
            batch["request"]["generationConfig"])
        online = prompts.online_request_from_batch(
            batch["key"], batch["request"])
        self.assertNotIn("media_resolution", online["config"])
        self.assertNotIn(
            "media_resolution", online["contents"][0]["parts"][1])

    def test_mixed_or_partial_ultra_high_placement_is_rejected(self):
        record = prompts.build_request(
            "stem0", self.IMAGES, prompt_type="osm_tags_farfield",
            media_resolution="MEDIA_RESOLUTION_ULTRA_HIGH",
            thinking_level="LOW")
        request = record["request"]
        del request["contents"][0]["parts"][0]["media_resolution"]
        with self.assertRaisesRegex(ValueError, "every image part"):
            prompts.online_request_from_batch(record["key"], request)


if __name__ == "__main__":
    unittest.main()
