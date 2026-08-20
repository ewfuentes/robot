import json
import unittest

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    semantic_audit as sa,
)


class FakeObs:
    def __init__(self, obs_id, primary="man_made=tower", confidence="medium",
                 description="a tower", extra=None, boxes=None):
        self.obs_id = obs_id
        self.primary_tag_key, self.primary_tag_value = primary.split("=")
        self.confidence = confidence
        self.description = description
        self.additional_tags = list((extra or {}).items())
        self.boxes = boxes or []


CLEAN = {"iou": 0.60, "inter_over_mask": 0.90, "inter_over_box": 0.65}
SUPERSET = {"iou": 0.30, "inter_over_mask": 1.00, "inter_over_box": 0.25}
CONTEXT = {"iou": 0.04, "inter_over_mask": 1.00, "inter_over_box": 0.04}
REJECT = {"iou": 0.00, "inter_over_mask": 0.50, "inter_over_box": 0.02}


def make_track(support_specs, birth=100, close_reason="starved",
               status="closed"):
    """support_specs: list of (keyframe, obs_id, metrics) -> minimal track
    artifact dict. One record per distinct keyframe, plus an unsupported
    record between birth and the end to exercise gap counting."""
    by_kf = {}
    for keyframe, obs_id, metrics in support_specs:
        by_kf.setdefault(keyframe, []).append(
            {"obs_id": obs_id, "class": "recorded_ignored",
             "box_window": [0, 0, 10, 10], **metrics})
    end = max(by_kf) if by_kf else birth
    records = []
    for keyframe in range(birth, end + 1):
        rec = {"keyframe": keyframe, "action": "continue_mask",
               "window_origin": [0.0, 0.0], "window_px": 1024,
               "mask_bbox_window": [1, 1, 9, 9]}
        if keyframe in by_kf:
            rec["supports"] = by_kf[keyframe]
        records.append(rec)
    return {"track_id": 7, "birth_keyframe": birth, "status": status,
            "close_reason": close_reason, "end_keyframe": end,
            "last_keyframe": end,
            "n_supported_keyframes": len(by_kf), "records": records}


class RunLengthEncodeTest(unittest.TestCase):
    def test_encodes_runs_in_order(self):
        self.assertEqual(sa.run_length_encode(["a", "a", "b", "a"]),
                         [("a", 2), ("b", 1), ("a", 1)])

    def test_empty(self):
        self.assertEqual(sa.run_length_encode([]), [])


class CollectEvidenceTest(unittest.TestCase):
    def test_splits_supports_context_and_rejects(self):
        cfg = sa.AuditConfig()
        track = make_track([(100, "a", CLEAN), (101, "b", CONTEXT),
                            (102, "c", REJECT)])
        obs = {k: FakeObs(k) for k in "abc"}
        supports, context = sa.collect_evidence(track, obs, cfg)
        self.assertEqual([e["obs"].obs_id for e in supports], ["a"])
        self.assertEqual([e["obs"].obs_id for e in context], ["b"])
        self.assertEqual(supports[0]["t"], 0)
        self.assertEqual(context[0]["t"], 1)


class TagVoteTableTest(unittest.TestCase):
    def test_counts_all_tags_and_confidences(self):
        cfg = sa.AuditConfig()
        track = make_track([(100, "a", CLEAN), (101, "b", CLEAN)])
        obs = {
            "a": FakeObs("a", primary="place=island", confidence="high",
                         extra={"name": "X", "distance_estimate": "over_10km",
                                "natural": "cliff"}),
            "b": FakeObs("b", primary="place=island", confidence="low"),
        }
        supports, _ = sa.collect_evidence(track, obs, cfg)
        rows = {r["tag"]: r for r in sa.tag_vote_table(supports)}
        island = rows["place=island"]
        self.assertEqual((island["total"], island["high"], island["low"],
                          island["as_primary"]), (2, 1, 1, 2))
        cliff = rows["natural=cliff"]
        self.assertEqual((cliff["total"], cliff["as_additional"]), (1, 1))
        self.assertNotIn("name=X", rows)
        self.assertNotIn("distance_estimate=over_10km", rows)


class SampleDescriptionsTest(unittest.TestCase):
    def _supports(self, n, tag_of=lambda i: "man_made=tower"):
        cfg = sa.AuditConfig()
        specs = [(100 + i, f"o{i}", CLEAN) for i in range(n)]
        obs = {f"o{i}": FakeObs(f"o{i}", primary=tag_of(i),
                                description=f"desc {i}") for i in range(n)}
        supports, _ = sa.collect_evidence(make_track(specs), obs, cfg)
        return supports

    def test_includes_first_last_and_respects_cap(self):
        supports = self._supports(30)
        picked = sa.sample_descriptions(supports, 10)
        self.assertEqual(len(picked), 10)
        self.assertEqual(picked[0]["t"], 0)
        self.assertEqual(picked[-1]["t"], 29)

    def test_rare_tag_is_added_beyond_stride(self):
        supports = self._supports(
            30, tag_of=lambda i: "man_made=crane" if i == 13 else
            "man_made=tower")
        # Place the rare tag off the stride: sampling 4 of 30 picks
        # t0, t10, t19, t29.
        picked = sa.sample_descriptions(supports, 4)
        tags = {e["obs"].primary_tag_value for e in picked}
        self.assertIn("crane", tags)

    def test_deterministic(self):
        supports = self._supports(30)
        a = [e["t"] for e in sa.sample_descriptions(supports, 10)]
        b = [e["t"] for e in sa.sample_descriptions(supports, 10)]
        self.assertEqual(a, b)


class SelectChipEntriesTest(unittest.TestCase):
    def test_first_last_budget_and_context_cap(self):
        cfg = sa.AuditConfig(max_support_chips=4, max_context_chips=1)
        specs = [(100 + i, f"o{i}", CLEAN) for i in range(20)]
        specs += [(130, "c0", CONTEXT), (131, "c1", CONTEXT)]
        obs = {f"o{i}": FakeObs(f"o{i}") for i in range(20)}
        obs["c0"] = FakeObs("c0")
        obs["c1"] = FakeObs("c1")
        supports, context = sa.collect_evidence(make_track(specs), obs, cfg)
        picked = sa.select_chip_entries(supports, context, cfg)
        ts = [e["t"] for e in picked]
        self.assertEqual(ts, sorted(ts))
        self.assertIn(0, ts)
        self.assertIn(19, ts)
        self.assertEqual(sum(1 for e in picked if e["is_context"]), 1)
        self.assertLessEqual(sum(1 for e in picked if not e["is_context"]), 4)

    def test_context_chips_cover_distinct_tags_before_repeating(self):
        # 5 fort context boxes with lower fill than a single tank box: the
        # tank group must still get a chip, or the model judges it from text
        # alone (the f0180 seawall-as-digester-tanks failure).
        cfg = sa.AuditConfig(max_support_chips=2, max_context_chips=2)
        specs = [(100, "s0", CLEAN), (101, "s1", CLEAN)]
        obs = {"s0": FakeObs("s0"), "s1": FakeObs("s1")}
        for i in range(5):
            metrics = dict(CONTEXT, inter_over_box=0.02)
            specs.append((110 + i, f"fort{i}", metrics))
            obs[f"fort{i}"] = FakeObs(f"fort{i}", primary="historic=fort")
        specs.append((120, "tank", dict(CONTEXT, inter_over_box=0.06)))
        obs["tank"] = FakeObs("tank", primary="man_made=storage_tank")
        supports, context = sa.collect_evidence(make_track(specs), obs, cfg)
        picked = sa.select_chip_entries(supports, context, cfg)
        ctx_tags = {e["obs"].primary_tag_value for e in picked
                    if e["is_context"]}
        self.assertEqual(ctx_tags, {"fort", "storage_tank"})

    def test_each_tag_run_gets_its_best_iou_chip(self):
        cfg = sa.AuditConfig(max_support_chips=4, max_context_chips=0)
        # Two runs: tower (t0-t9), crane (t10-t19); crane's best iou at t15.
        specs = []
        for i in range(20):
            metrics = dict(CLEAN)
            metrics["iou"] = 0.90 if i == 15 else metrics["iou"]
            specs.append((100 + i, f"o{i}", metrics))
        obs = {f"o{i}": FakeObs(
            f"o{i}", primary="man_made=crane" if i >= 10 else "man_made=tower")
            for i in range(20)}
        supports, context = sa.collect_evidence(make_track(specs), obs, cfg)
        picked = sa.select_chip_entries(supports, context, cfg)
        self.assertIn(15, [e["t"] for e in picked])


class DossierTest(unittest.TestCase):
    def _dossier(self):
        cfg = sa.AuditConfig()
        specs = [(100, "a", CLEAN), (101, "b", SUPERSET), (103, "c", CLEAN),
                 (104, "x", CONTEXT)]
        obs = {
            "a": FakeObs("a", primary="man_made=water_tower", confidence="high",
                         description="white cylinder",
                         extra={"distance_estimate": "2km_to_10km"}),
            "b": FakeObs("b", primary="man_made=lighthouse",
                         description="white tower",
                         extra={"distance_estimate": "over_10km",
                                "name": "Foo Light"}),
            "c": FakeObs("c", primary="man_made=water_tower",
                         description="banded standpipe"),
            "x": FakeObs("x", primary="landuse=industrial",
                         description="tank cluster"),
        }
        return sa.build_dossier(make_track(specs), obs, cfg)

    def test_counts_and_relative_time(self):
        d = self._dossier()
        self.assertEqual(d["n_supports"], 3)
        self.assertEqual(d["lifetime"], 5)
        # t2 has no support; t4's only detection is context.
        self.assertEqual(d["n_gap_keyframes"], 2)
        self.assertEqual(d["primary_tag_rle"],
                         [("man_made=water_tower", 1),
                          ("man_made=lighthouse", 1),
                          ("man_made=water_tower", 1)])
        self.assertEqual(d["distance_rle"][-1], ("unreported", 1))

    def test_rendered_text_is_scrubbed_and_complete(self):
        d = self._dossier()
        text = sa.render_dossier_text(d)
        # No absolute keyframe ids leak through - time is relative only.
        self.assertNotIn("f010", text)
        self.assertNotIn("100", text)
        for needle in ["lifetime: 5 keyframes", "names reported, with the detector's confidence in each name: "
            "'Foo Light' x1 (1 medium)",
                       "run-length encoded", "man_made=water_tower x1",
                       "banded standpipe", "tank cluster", "mask fills 4%"]:
            self.assertIn(needle, text)

    def test_alive_track_close_text(self):
        cfg = sa.AuditConfig()
        track = make_track([(100, "a", CLEAN)], status="alive",
                           close_reason="")
        d = sa.build_dossier(track, {"a": FakeObs("a")}, cfg)
        self.assertIn("still active", d["close_text"])


class EvidenceTest(unittest.TestCase):
    def _build(self, name_seq):
        """One support per keyframe, names taken from name_seq."""
        cfg = sa.AuditConfig()
        specs, obs = [], {}
        for i, nm in enumerate(name_seq):
            specs.append((100 + i, f"o{i}", CLEAN))
            extra = {"name": nm} if nm else {}
            obs[f"o{i}"] = FakeObs(f"o{i}", confidence="high", extra=extra)
        track = make_track(specs)
        dossier = sa.build_dossier(track, obs, cfg)
        track["n_supported_keyframes"] = dossier["n_supports"]
        return sa.build_evidence(track, dossier, pano_w=1000)

    def test_counts_travel_with_the_record(self):
        ev = self._build(["A"] * 8)
        self.assertEqual(ev["n_supports"], 8)
        self.assertEqual(ev["lifetime_keyframes"], 8)
        self.assertEqual(ev["support_density"], 1.0)
        self.assertEqual(ev["confidence_counts"]["high"], 8)
        self.assertGreater(ev["median_iou"], 0)

    def test_unanimous_name_is_not_contested(self):
        ev = self._build(["A"] * 5)
        self.assertFalse(ev["name_contested"])
        self.assertEqual(ev["name_top_share"], 1.0)
        self.assertEqual(ev["n_distinct_names"], 1)

    def test_split_names_are_contested(self):
        # T4 shape: a plurality name that is still a minority overall.
        ev = self._build(["A"] * 4 + ["B"] * 3 + ["C"] * 3)
        self.assertTrue(ev["name_contested"])
        self.assertLess(ev["name_top_share"], 0.5)
        self.assertEqual(ev["n_distinct_names"], 3)

    def test_single_stray_name_is_flagged_by_volume_not_share(self):
        # T5 shape: 1 name vote among 4 supports - share is 1.0, so the
        # contested flag stays false; n_named_supports is what reveals it.
        ev = self._build([None, None, None, "A"])
        self.assertFalse(ev["name_contested"])
        self.assertEqual(ev["n_named_supports"], 1)
        self.assertEqual(ev["n_supports"], 4)

    def test_azimuth_span_is_wrap_safe(self):
        # Masks marching across the wrap must not report a ~360 deg span.
        cfg = sa.AuditConfig()
        specs = [(100 + i, f"o{i}", CLEAN) for i in range(3)]
        obs = {f"o{i}": FakeObs(f"o{i}") for i in range(3)}
        track = make_track(specs)
        for rec, x in zip(track["records"], (980, 0, 20)):
            rec["window_origin"] = [x, 0.0]
            rec["mask_bbox_window"] = [0, 1, 10, 9]
        dossier = sa.build_dossier(track, obs, cfg)
        ev = sa.build_evidence(track, dossier, pano_w=1000)
        self.assertLess(ev["camera_azimuth_span_deg"], 30.0)


class SchemaTest(unittest.TestCase):
    def test_schema_is_inlined_and_fully_required(self):
        schema = sa.get_audit_schema()
        blob = json.dumps(schema)
        self.assertNotIn("$ref", blob)
        self.assertNotIn("title", blob)

        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" in node:
                    self.assertEqual(set(node["required"]),
                                     set(node["properties"].keys()))
                for v in node.values():
                    check(v)
            elif isinstance(node, list):
                for v in node:
                    check(v)
        check(schema)
        self.assertIn("landmark_kind", schema["properties"])


class RequestTest(unittest.TestCase):
    def test_request_structure_and_interleaving(self):
        cfg = sa.AuditConfig()
        req = sa.build_request("T7", "dossier text",
                               [("cap1", "b64a"), ("cap2", "b64b")], cfg)
        self.assertEqual(req["key"], "T7")
        parts = req["request"]["contents"][0]["parts"]
        kinds = ["text" if "text" in p else "image" for p in parts]
        self.assertEqual(kinds, ["text", "text", "image", "text", "image"])
        self.assertEqual(parts[2]["inline_data"]["data"], "b64a")
        gen = req["request"]["generationConfig"]
        self.assertEqual(gen["responseMimeType"], "application/json")
        self.assertIn("responseSchema", gen)
        self.assertEqual(req["request"]["systemInstruction"]["parts"][0]
                         ["text"], sa.SYSTEM_PROMPT)


class ParseResultTest(unittest.TestCase):
    def test_roundtrip(self):
        audit = sa.TrackAudit(
            landmark_kind="fixed_structure", single_object=True,
            valid_segments=[sa.Segment(start_t=0, end_t=10)],
            verdict="keep", drop_reason="none",
            primary_object=sa.PrimaryObject(
                tags=[sa.WeightedTag(tag="man_made=water_tower", weight=0.6)],
                name_candidates=[sa.NameCandidate(
                    name="Foo Light", weight=0.7,
                    basis="reported_by_detections")],
                name_aliases=[], description="white cylinder",
                distinctive_features=["banded"], extent="point_like"),
            strike_votes=[], secondary_objects=[], confidence="high",
            unresolved="")
        record = {"key": "T7", "response": {"candidates": [{"content": {
            "parts": [{"text": audit.model_dump_json()}]}}]}}
        key, parsed, err = sa.parse_result_line(record)
        self.assertEqual(key, "T7")
        self.assertIsNone(err)
        self.assertEqual(parsed["primary_object"]["tags"][0]["tag"],
                         "man_made=water_tower")

    def test_error_and_malformed_lines(self):
        key, parsed, err = sa.parse_result_line(
            {"key": "T1", "error": "boom", "response": None})
        self.assertEqual((key, parsed), ("T1", None))
        self.assertIn("boom", err)
        key, parsed, err = sa.parse_result_line(
            {"key": "T2", "response": {"candidates": [{"content": {
                "parts": [{"text": "not json"}]}}]}})
        self.assertIsNone(parsed)
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()


class NameConfidenceTest(unittest.TestCase):
    """The dossier must carry the detector's confidence in each NAME.

    Every identity tag reaches the auditor split high/medium/low, but `name`
    is in NON_IDENTITY_TAG_KEYS and so is excluded from the tag table. The
    extraction prompt tells the detector to express naming doubt through
    `confidence`, so a dossier that drops it discards exactly that signal --
    which is how a name asserted once at medium confidence reached the
    matcher weighted 0.8 (boston_harbor_leg1, T224).
    """

    def _dossier(self, specs):
        track = make_track([(kf, oid, CLEAN) for kf, oid, _, _ in specs])
        obs = {oid: FakeObs(oid, confidence=conf, extra={"name": name})
               for _, oid, name, conf in specs if name}
        obs.update({oid: FakeObs(oid, confidence=conf)
                    for _, oid, name, conf in specs if not name})
        return sa.build_dossier(track, obs, sa.AuditConfig())

    def test_confidence_split_recorded_and_rendered(self):
        d = self._dossier([
            (101, "a", "Georges Island", "medium"),
            (102, "b", "Custom House Tower", "high"),
            (103, "c", "Custom House Tower", "high"),
            (104, "d", "Custom House Tower", "low"),
        ])
        self.assertEqual(d["name_confidence"]["Georges Island"],
                         {"high": 0, "medium": 1, "low": 0})
        self.assertEqual(d["name_confidence"]["Custom House Tower"],
                         {"high": 2, "medium": 0, "low": 1})
        text = sa.render_dossier_text(d)
        self.assertIn("'Custom House Tower' x3 (2 high, 1 low)", text)
        self.assertIn("'Georges Island' x1 (1 medium)", text)

    def test_unnamed_track_still_says_none(self):
        d = self._dossier([(101, "a", "", "high"), (102, "b", "", "high")])
        self.assertEqual(d["name_confidence"], {})
        self.assertIn("names reported: (none)", sa.render_dossier_text(d))

    def test_prompt_calibrates_weight_against_support(self):
        # The measured failure was a name with 1 of 64 detections emitted at
        # weight 0.8; the prompt must state the ceiling explicitly.
        self.assertIn("CALIBRATE THE WEIGHT AGAINST THE SUPPORT",
                      sa.SYSTEM_PROMPT)
        self.assertIn("below 0.3", sa.SYSTEM_PROMPT)
