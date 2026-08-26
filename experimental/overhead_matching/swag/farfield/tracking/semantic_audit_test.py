"""Library tests: dossier/evidence construction from synthetic tracks and
observations, chip/description selection, result-line parsing."""

import hashlib
import json
import unittest

from google.genai import types as genai_types

from experimental.overhead_matching.swag.farfield import dataset
from experimental.overhead_matching.swag.farfield.tracking import (
    semantic_audit as sa,
    track_builder as tb,
)

PANO_W = 1024
LEGACY_SYSTEM_PROMPT_SHA256 = (
    "f40a8e8f886219da3bbce5111910cf9f6aa05de503dccb2476dfbb17c4949112")
LEGACY_AUDIT_SCHEMA_SHA256 = (
    "92eb7397371304c8cd776b27f8f7fdc02ca4f7107ad64822cf303e29972a23cd")


def json_digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_cfg(**overrides):
    values = dict(
        min_supports=2, max_support_chips=6, max_context_chips=2,
        max_description_samples=10, chip_height_px=320,
        thinking_level="HIGH",
        classifier=tb.TrackBuilderConfig(reference_pano_width=PANO_W))
    values.update(overrides)
    return sa.AuditConfig(**values)


def make_obs(obs_id, primary=("man_made", "tower"), name="",
             confidence="high", description="a tower", frame_idx=0):
    key = dataset.ObservationKey(
        dataset="synthetic",
        frame_landmarks_version="semantic-audit-test",
        frame_landmarks_content_digest="0" * 64,
        local_obs_id=obs_id)
    return dataset.Observation(
        key=key, obs_id=obs_id, local_obs_id=obs_id,
        pano_id=f"f{frame_idx:04d}", frame_idx=frame_idx,
        landmark_idx=0, embedding_id="",
        primary_tag_key=primary[0], primary_tag_value=primary[1],
        additional_tags=[["name", name]] if name else [],
        confidence=confidence, description=description, boxes=[],
        seam_merged=False, bearing_camera_cw_deg=0.0, elevation_deg=0.0,
        angular_width_deg=5.0)


def support(obs_id, iou, iom, iob, box=(40.0, 30.0, 80.0, 60.0)):
    return {"obs_id": obs_id, "class": "recorded-at-run-time",
            "box_window": list(box), "iou": iou,
            "inter_over_mask": iom, "inter_over_box": iob}


def record(keyframe, supports=(), mask_bbox=(40, 30, 80, 60),
           origin=(0.0, 0), action="continue_mask"):
    return {"keyframe": keyframe, "action": action,
            "window_origin": list(origin), "window_px": 256,
            "mask_area": 100,
            "mask_bbox_window": list(mask_bbox) if mask_bbox else None,
            "supports": list(supports)}


def make_track(track_id, records, birth_keyframe=None, close_reason="starved",
               status="closed"):
    supported = {r["keyframe"] for r in records if any(
        s["iou"] >= 0.45 for s in r.get("supports", []))}
    return {
        "track_id": track_id,
        "birth_obs_id": "b0",
        "birth_keyframe": (birth_keyframe if birth_keyframe is not None
                           else records[0]["keyframe"]),
        "status": status, "close_reason": close_reason,
        "end_keyframe": max(r["keyframe"] for r in records),
        "last_keyframe": max(r["keyframe"] for r in records),
        "modal_label": "man_made=tower",
        "n_supported_keyframes": len(supported),
        "records": records,
    }


# Classifier landmarks (see track_builder.classify_support):
CLEAN = dict(iou=0.8, iom=0.9, iob=0.9)         # continue_clean
CONTEXT = dict(iou=0.05, iom=0.9, iob=0.05)     # contains mask, incoherent
NONE = dict(iou=0.0, iom=0.0, iob=0.0)          # rejected outright


class CollectEvidenceTest(unittest.TestCase):
    def test_splits_support_context_none_under_recorded_classifier(self):
        cfg = make_cfg()
        obs = {oid: make_obs(oid) for oid in ("a", "b", "c")}
        track = make_track(1, [
            record(0, action="birth", supports=[]),
            record(1, supports=[
                support("a", **CLEAN), support("b", **CONTEXT),
                support("c", **NONE)]),
        ])
        supports, context = sa.collect_evidence(track, obs, cfg)
        self.assertEqual([e["support"]["obs_id"] for e in supports], ["a"])
        self.assertEqual([e["support"]["obs_id"] for e in context], ["b"])

    def test_unknown_obs_ids_are_a_hard_error(self):
        cfg = make_cfg()
        track = make_track(1, [record(0, supports=[support("gone", **CLEAN)])])
        with self.assertRaisesRegex(ValueError, "unknown observation 'gone'"):
            sa.collect_evidence(track, {}, cfg)

    def test_time_indices_are_relative_to_birth(self):
        cfg = make_cfg()
        obs = {"a": make_obs("a")}
        track = make_track(1, [
            record(10, action="birth"),
            record(12, supports=[support("a", **CLEAN)]),
        ])
        supports, _ = sa.collect_evidence(track, obs, cfg)
        self.assertEqual(supports[0]["t"], 2)
        self.assertEqual(supports[0]["keyframe"], 12)


class DossierTest(unittest.TestCase):
    def _track_and_obs(self):
        obs = {
            "b0": make_obs("b0", confidence="high",
                           description="the founding tower", frame_idx=10),
            "o1": make_obs("o1", name="Graves Light", confidence="high",
                           frame_idx=11),
            "o2": make_obs("o2", name="Graves Light", confidence="high",
                           frame_idx=12),
            "o3": make_obs("o3", name="Graves Light", confidence="medium",
                           frame_idx=13),
            "o4": make_obs("o4", primary=("man_made", "lighthouse"),
                           frame_idx=14),
        }
        track = make_track(7, [
            record(10, action="birth"),
            record(11, supports=[support("o1", **CLEAN)]),
            record(12, supports=[support("o2", **CLEAN)],
                   action="reanchor_clean"),
            record(13, supports=[support("o3", **CLEAN)]),
            record(14, supports=[support("o4", **CLEAN)]),
        ])
        return track, obs

    def test_dossier_counts_rle_and_name_confidence(self):
        track, obs = self._track_and_obs()
        d = sa.build_dossier(track, obs, make_cfg())
        self.assertEqual(d["n_supports"], 4)
        self.assertEqual(d["n_evidence_detections"], 5)
        self.assertEqual(d["lifetime"], 5)
        self.assertEqual(d["n_reanchors"], 1)
        self.assertEqual(d["n_gap_keyframes"], 0)
        self.assertEqual(d["primary_tag_rle"],
                         [("man_made=tower", 4), ("man_made=lighthouse", 1)])
        self.assertEqual(d["name_votes"], [("Graves Light", 3)])
        self.assertEqual(d["name_confidence"]["Graves Light"],
                         {"high": 2, "medium": 1, "low": 0})
        # `name` is not an identity tag: only the primaries reach the table.
        self.assertEqual({r["tag"] for r in d["tag_table"]},
                         {"man_made=tower", "man_made=lighthouse"})
        tower = next(r for r in d["tag_table"] if r["tag"] == "man_made=tower")
        self.assertEqual((tower["total"], tower["as_primary"], tower["high"],
                          tower["medium"]), (4, 4, 3, 1))

    def test_dossier_text_renders_every_section(self):
        track, obs = self._track_and_obs()
        d = sa.build_dossier(track, obs, make_cfg())
        text = sa.render_dossier_text(d)
        self.assertIn("TRACK EVIDENCE", text)
        self.assertIn("'Graves Light' x3 (2 high, 1 medium)", text)
        self.assertIn("man_made=tower x4", text)
        self.assertIn("1 founding detection at t0 + 4 post-birth", text)
        self.assertIn("unreported x5", text)  # distance_estimate RLE
        self.assertIn("the founding tower", text)
        self.assertIn(sa.QUESTIONS_TEXT, text)

    def test_evidence_math(self):
        track, obs = self._track_and_obs()
        cfg = make_cfg()
        d = sa.build_dossier(track, obs, cfg)
        ev = sa.build_evidence(track, d, PANO_W)
        self.assertEqual(ev["n_supports"], 4)
        self.assertEqual(ev["lifetime_keyframes"], 5)
        self.assertAlmostEqual(ev["support_density"], 4 / 5)
        self.assertEqual(ev["n_reanchors"], 1)
        self.assertAlmostEqual(ev["tag_top_share"], 4 / 5)
        self.assertEqual(ev["name_votes"], {"Graves Light": 3})
        self.assertEqual(ev["n_named_supports"], 3)
        self.assertAlmostEqual(ev["name_top_share"], 1.0)
        self.assertEqual(ev["name_margin"], 3.0)
        self.assertFalse(ev["name_contested"])
        self.assertEqual(ev["confidence_counts"],
                         {"high": 4, "medium": 1, "low": 0})
        # All mask boxes identical -> no azimuth sweep.
        self.assertEqual(ev["camera_azimuth_span_deg"], 0.0)

    def test_founder_is_always_a_t0_chip(self):
        track, obs = self._track_and_obs()
        dossier = sa.build_dossier(track, obs, make_cfg())
        founders = [
            entry for entry in dossier["chip_entries"]
            if entry.get("is_founding")]
        self.assertEqual(len(founders), 1)
        self.assertEqual(founders[0]["obs"].obs_id, "b0")
        self.assertIn("FOUNDING", sa.chip_caption(founders[0], 1))

    def test_missing_founder_is_a_hard_error(self):
        track, obs = self._track_and_obs()
        del obs["b0"]
        with self.assertRaisesRegex(ValueError, "birth references unknown"):
            sa.build_dossier(track, obs, make_cfg())

    def test_contested_names_and_azimuth_span(self):
        obs = {
            "b0": make_obs("b0", frame_idx=0),
            "o1": make_obs("o1", name="Fort Warren", frame_idx=1),
            "o2": make_obs("o2", name="Fort Warren", frame_idx=2),
            "o3": make_obs("o3", name="Fort Independence", frame_idx=3),
            "o4": make_obs("o4", name="Fort Independence", frame_idx=4),
        }
        track = make_track(3, [
            record(0, action="birth"),
            record(1, supports=[support("o1", **CLEAN)],
                   mask_bbox=(100, 10, 110, 20)),
            record(2, supports=[support("o2", **CLEAN)],
                   mask_bbox=(150, 10, 160, 20)),
            record(3, supports=[support("o3", **CLEAN)],
                   mask_bbox=(200, 10, 210, 20)),
            record(4, supports=[support("o4", **CLEAN)],
                   mask_bbox=(200, 10, 210, 20)),
        ], birth_keyframe=0)
        ev = sa.build_evidence(track, sa.build_dossier(track, obs, make_cfg()),
                               PANO_W)
        self.assertTrue(ev["name_contested"])  # 2 vs 2 split
        self.assertEqual(ev["name_margin"], 1.0)
        self.assertAlmostEqual(ev["camera_azimuth_span_deg"],
                               (205 - 60) / PANO_W * 360.0)


class SelectionTest(unittest.TestCase):
    def test_sample_descriptions_covers_every_tag_within_cap(self):
        obs, supports = {}, []
        for i in range(12):
            primary = ("natural", "cliff") if i == 5 else ("man_made", "tower")
            oid = f"o{i}"
            obs[oid] = make_obs(oid, primary=primary, frame_idx=i)
            supports.append({"t": i, "keyframe": i, "obs": obs[oid],
                             "support": support(oid, **CLEAN),
                             "rec": record(i)})
        picked = sa.sample_descriptions(supports, max_samples=4)
        self.assertLessEqual(len(picked), 4 + 1)  # stride + rare-tag rescue
        picked_tags = {f"{e['obs'].primary_tag_key}="
                       f"{e['obs'].primary_tag_value}" for e in picked}
        self.assertIn("natural=cliff", picked_tags)
        self.assertEqual([e["t"] for e in picked], sorted(e["t"]
                                                          for e in picked))
        # First and last are always present.
        self.assertEqual(picked[0]["t"], 0)
        self.assertEqual(picked[-1]["t"], 11)

    def test_select_chip_entries_first_last_cap_and_context(self):
        cfg = make_cfg(max_support_chips=3, max_context_chips=1)
        supports, context = [], []
        for i in range(8):
            oid = f"s{i}"
            supports.append({"t": i, "keyframe": i, "obs": make_obs(oid),
                             "support": support(oid, iou=0.5 + 0.01 * i,
                                                iom=0.9, iob=0.9),
                             "rec": record(i)})
        for i, tag in enumerate([("building", "yes"), ("building", "yes"),
                                 ("natural", "coastline")]):
            oid = f"c{i}"
            context.append({"t": 20 + i, "keyframe": 20 + i,
                            "obs": make_obs(oid, primary=tag),
                            "support": support(oid, iou=0.05, iom=0.9,
                                               iob=0.02 + 0.01 * i),
                            "rec": record(20 + i)})
        entries = sa.select_chip_entries(supports, context, cfg)
        ts = [e["t"] for e in entries]
        self.assertEqual(ts, sorted(ts))
        support_ts = [e["t"] for e in entries if not e["is_context"]]
        self.assertLessEqual(len(support_ts), cfg.max_support_chips)
        self.assertIn(0, support_ts)   # first support
        self.assertIn(7, support_ts)   # last support
        ctx_entries = [e for e in entries if e["is_context"]]
        self.assertEqual(len(ctx_entries), 1)
        # The doubly-claimed tag wins the single context slot, represented by
        # its largest mask fill.
        self.assertEqual(ctx_entries[0]["obs"].primary_tag_key, "building")
        self.assertEqual(ctx_entries[0]["t"], 21)


class MergeTracksTest(unittest.TestCase):
    def test_merges_ranges_and_maps_range_names(self):
        artifacts = {
            "legA": {"tracks": [make_track(1, [record(0)])]},
            "legB": {"tracks": [make_track(2, [record(0)])]},
        }
        tracks, ranges = sa.merge_tracks(artifacts)
        self.assertEqual(set(tracks), {1, 2})
        self.assertEqual(ranges, {1: "legA", 2: "legB"})

    def test_duplicate_track_id_across_ranges_is_refused(self):
        artifacts = {
            "legA": {"tracks": [make_track(1, [record(0)])]},
            "legB": {"tracks": [make_track(1, [record(0)])]},
        }
        with self.assertRaises(SystemExit) as ctx:
            sa.merge_tracks(artifacts)
        self.assertIn("track_id", str(ctx.exception))


class SchemaTest(unittest.TestCase):
    def test_schema_is_inlined_and_required_everywhere(self):
        schema = sa.get_provider_audit_schema()
        self.assertNotIn("$ref", json.dumps(schema))
        self.assertEqual(set(schema), {"anyOf"})
        for variant in schema["anyOf"]:
            self.assertEqual(set(variant["required"]),
                             set(variant["properties"].keys()))
            primary = variant["properties"]["primary_object"]
            self.assertEqual(set(primary["required"]),
                             set(primary["properties"].keys()))

    def test_prompt_and_schema_carry_the_same_decision_contract(self):
        schema = sa.get_provider_audit_schema()
        self.assertIn(sa.PROVIDER_DECISION_CONTRACT, sa.SYSTEM_PROMPT)
        for variant in schema["anyOf"]:
            for field in ("decision", "valid_segments"):
                self.assertEqual(
                    variant["properties"][field]["description"],
                    variant["description"])
        for language in (
                "keep_single means one physical identity",
                "keep_partial_identity_switch only means a true switch",
                "stable wrong-target",
                "multi-object switch",
                "Drop valid_segments may be empty or nonempty"):
            self.assertIn(language, sa.PROVIDER_DECISION_CONTRACT)

    def test_schema_uses_only_supported_variant_constraints(self):
        schema = sa.get_provider_audit_schema()
        encoded = json.dumps(schema)
        self.assertNotIn('"oneOf"', encoded)
        self.assertNotIn('"const"', encoded)
        self.assertEqual(set(schema), {"anyOf"})
        accepted, dropped = schema["anyOf"]
        for variant in (accepted, dropped):
            self.assertNotIn("verdict", variant["properties"])
            self.assertNotIn("single_object", variant["properties"])
            self.assertNotIn("drop_reason", variant["properties"])
        self.assertEqual(
            accepted["properties"]["valid_segments"]["minItems"], 1)
        self.assertEqual(
            set(accepted["properties"]["decision"]["enum"]),
            {"keep_single", "keep_partial_identity_switch"})
        self.assertNotIn(
            "minItems", dropped["properties"]["valid_segments"])
        self.assertEqual(
            set(dropped["properties"]["decision"]["enum"]),
            {decision for decision in sa.PROVIDER_DECISION_TO_CANONICAL
             if decision.startswith("drop_")})
        self.assertEqual(
            (set(accepted["properties"]["decision"]["enum"])
             | set(dropped["properties"]["decision"]["enum"])),
            set(sa.PROVIDER_DECISION_TO_CANONICAL))

        # Exercise the repository's pinned Gemini SDK dialect rather than
        # assuming arbitrary JSON Schema keywords are provider-compatible.
        provider_schema = genai_types.Schema.model_validate(schema)
        self.assertEqual(len(provider_schema.any_of), 2)
        provider_json = provider_schema.model_dump(
            by_alias=True, exclude_none=True)
        self.assertIn("anyOf", provider_json)
        self.assertNotIn("oneOf", provider_json)

    def test_contract_changes_both_sealed_request_digests(self):
        self.assertNotEqual(
            hashlib.sha256(sa.SYSTEM_PROMPT.encode()).hexdigest(),
            LEGACY_SYSTEM_PROMPT_SHA256)
        self.assertNotEqual(
            json_digest(sa.get_provider_audit_schema()),
            LEGACY_AUDIT_SCHEMA_SHA256)


def valid_audit_payload(**overrides):
    payload = sa.TrackAudit(
        landmark_kind="fixed_structure",
        single_object=True,
        valid_segments=[sa.Segment(start_t=0, end_t=4)],
        verdict="keep",
        drop_reason="none",
        primary_object=sa.PrimaryObject(
            tags=[sa.WeightedTag(tag="man_made=lighthouse", weight=0.9)],
            name_candidates=[sa.NameCandidate(
                name="Graves Light", weight=0.8, basis="both")],
            name_aliases=[],
            description="white conical masonry tower",
            distinctive_features=["black lantern"],
            extent="point_like"),
        strike_votes=[sa.StrikeVote(t=2, reason="different building")],
        secondary_objects=[],
        confidence="high",
        unresolved="").model_dump()
    payload.update(overrides)
    return payload


def provider_audit_payload(decision="keep_single", **overrides):
    canonical = valid_audit_payload()
    payload = {
        key: value for key, value in canonical.items()
        if key not in {"single_object", "verdict", "drop_reason"}
    }
    payload["decision"] = decision
    payload.update(overrides)
    return payload


class CanonicalizeProviderAuditTest(unittest.TestCase):
    def test_every_decision_has_one_deterministic_canonical_mapping(self):
        for decision, expected in sa.PROVIDER_DECISION_TO_CANONICAL.items():
            with self.subTest(decision=decision):
                segments = ([{"start_t": 0, "end_t": 4}]
                            if decision.startswith("keep_") else [])
                provider = sa.ProviderTrackAudit.model_validate(
                    provider_audit_payload(
                        decision, valid_segments=segments))
                canonical = sa.canonicalize_provider_audit(provider)
                self.assertEqual(
                    (canonical["verdict"], canonical["single_object"],
                     canonical["drop_reason"]),
                    expected)
                self.assertEqual(canonical["valid_segments"], segments)
                self.assertNotIn("decision", canonical)


def result_line(key, payload):
    return {"key": key,
            "response": {"candidates": [{"content": {"parts": [
                {"text": json.dumps(payload)}]}}]}}


class ParseResultLineTest(unittest.TestCase):
    def test_round_trip(self):
        payload = valid_audit_payload()
        key, audit, err = sa.parse_result_line(result_line("T7", payload))
        self.assertEqual((key, err), ("T7", None))
        self.assertEqual(audit, payload)

    def test_error_line(self):
        key, audit, err = sa.parse_result_line({"key": "T9", "error": "quota"})
        self.assertEqual((key, audit, err), ("T9", None, "quota"))

    def test_unparseable_payload_reports_not_raises(self):
        key, audit, err = sa.parse_result_line(
            {"key": "T3", "response": {"candidates": [{"content": {"parts": [
                {"text": "not json {"}]}}]}})
        self.assertEqual(key, "T3")
        self.assertIsNone(audit)
        self.assertIn("JSONDecodeError", err)

    def test_legacy_single_name_is_rejected(self):
        payload = valid_audit_payload()
        payload["primary_object"].pop("name_candidates")
        payload["primary_object"]["name"] = "Old Name"
        key, audit, err = sa.parse_result_line(result_line("T1", payload))
        self.assertEqual(key, "T1")
        self.assertIsNone(audit)
        self.assertIn("ValidationError", err)


if __name__ == "__main__":
    unittest.main()
