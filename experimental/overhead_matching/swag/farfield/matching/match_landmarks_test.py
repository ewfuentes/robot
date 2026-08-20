"""Tests for the matcher, on a tiny synthetic run. No network anywhere:
requests are built with --build_only, results are fabricated in the shape
vertex_batch_manager writes, and aggregation runs with --aggregate_only."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import msgspec
import shapely

from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.localization import structs
from experimental.overhead_matching.swag.farfield.matching import (
    match_landmarks as ml,
)

ANCHOR_LAT, ANCHOR_LON = 42.35, -71.05


def write_feather(path: Path) -> Path:
    """Five rows: one unique signature, one two-row signature, one ENC row,
    one row with no far-field tags at all."""
    frame = schema.build_frame(
        ids=["('node', 101)", "('node', 102)", "('node', 103)",
             "('lights', 7)", "('node', 999)"],
        geometries=[
            shapely.Point(ANCHOR_LON + 0.001, ANCHOR_LAT + 0.001),
            shapely.Point(ANCHOR_LON + 0.002, ANCHOR_LAT + 0.001),
            shapely.Point(ANCHOR_LON + 0.002, ANCHOR_LAT + 0.002),
            shapely.Point(ANCHOR_LON - 0.001, ANCHOR_LAT - 0.001),
            shapely.Point(ANCHOR_LON, ANCHOR_LAT),
        ],
        landmark_types=["osm", "osm", "osm", "enc", "osm"],
        tags=[
            {"man_made": "lighthouse", "name": "Graves Light"},
            {"man_made": "tower"},
            {"man_made": "tower"},
            {"object_class": "LIGHTS"},
            {"addr:street": "nothing far-field"},
        ],
    )
    frame.to_feather(path)
    return path


def track(track_id: int, keyframes) -> dict:
    return {
        "track_id": track_id,
        "birth_keyframe": min(keyframes),
        "close_reason": "end_of_range",
        "records": [
            {"keyframe": kf, "action": "propagate",
             "window_origin": [0, 0],
             "mask_bbox_window": [30, 10, 34, 14],
             "supports": []}
            for kf in keyframes
        ],
    }


def audit(verdict="keep", name="Graves Light") -> dict:
    return {
        "verdict": verdict,
        "landmark_kind": "fixed_structure",
        "valid_segments": [{"start_t": 0, "end_t": 10}],
        "unresolved": "possibly a monument instead",
        "primary_object": {
            "tags": [{"tag": "man_made=lighthouse", "weight": 0.9},
                     {"tag": "man_made=tower", "weight": 0.4}],
            "name_candidates": [
                {"name": name, "weight": 0.8, "basis": "both"}],
            "name_aliases": [],
            "description": "white conical tower on a rock",
            "distinctive_features": ["red lantern room"],
            "extent": "point_like",
        },
    }


def write_run(run_dir: Path, split_ranges=False) -> None:
    """tracks_*.json (+ optionally a second range file) and a semantic audit
    for tracks 1 (keep), 2 (keep), 3 (drop). Track 4 exists but was never
    audited."""
    run_dir.mkdir(parents=True, exist_ok=True)
    first = {"range": {"name": "seg0"},
             "tracks": [track(1, [0, 1, 2]), track(2, [1, 2, 3])]}
    second = {"range": {"name": "seg1"},
              "tracks": [track(3, [4, 5]), track(4, [4, 5])]}
    if split_ranges:
        (run_dir / "tracks_seg0.json").write_text(json.dumps(first))
        (run_dir / "tracks_seg1.json").write_text(json.dumps(second))
    else:
        merged = {"range": {"name": "seg0"},
                  "tracks": first["tracks"] + second["tracks"]}
        (run_dir / "tracks_seg0.json").write_text(json.dumps(merged))

    audit_dir = run_dir / "semantic_audit"
    audit_dir.mkdir(exist_ok=True)
    meta = {}
    with open(audit_dir / "results.jsonl", "w") as f:
        for tid, payload in ((1, audit()),
                             (2, audit(name="Customs Tower")),
                             (3, audit(verdict="drop"))):
            key = f"T{tid}"
            meta[key] = {"track_id": tid, "birth_keyframe": 0,
                         "n_supports": 3, "chips": []}
            f.write(json.dumps({
                "key": key,
                "response": {"candidates": [{"content": {"parts": [
                    {"text": json.dumps(payload)}]}}]},
            }) + "\n")
    (audit_dir / "audit_meta.json").write_text(json.dumps(meta))


def run_main(argv):
    old = sys.argv
    sys.argv = ["match_landmarks"] + argv
    try:
        ml.main()
    finally:
        sys.argv = old


class LoadTracksTest(unittest.TestCase):
    def test_loads_every_range_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_run(run_dir, split_ranges=True)
            tracks, range_by_track = ml.load_tracks(run_dir)
            self.assertEqual(set(tracks), {1, 2, 3, 4})
            self.assertEqual(range_by_track[1], "seg0")
            self.assertEqual(range_by_track[3], "seg1")

    def test_duplicate_track_id_across_ranges_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            artifact = {"range": {"name": "seg0"}, "tracks": [track(1, [0])]}
            (run_dir / "tracks_a.json").write_text(json.dumps(artifact))
            (run_dir / "tracks_b.json").write_text(json.dumps(
                {"range": {"name": "seg1"}, "tracks": [track(1, [1])]}))
            with self.assertRaises(SystemExit):
                ml.load_tracks(run_dir)

    def test_no_tracks_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                ml.load_tracks(Path(tmp))


class QueryBundlesTest(unittest.TestCase):
    def test_one_entry_per_audited_track_drop_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_run(run_dir, split_ranges=True)
            from experimental.overhead_matching.swag.farfield.calibration \
                import audit_io
            tracks, _ = ml.load_tracks(run_dir)
            audits = audit_io.load_audits(run_dir)
            queries = ml.query_bundles(tracks, audits)
            # 1 and 2 audited keep; 3 audited drop; 4 never audited.
            self.assertEqual(set(queries), {"T1", "T2"})

    def test_query_block_carries_the_audit_uncertainty(self):
        block = ml.format_query(audit())
        self.assertIn("tags: man_made=lighthouse (0.90); "
                      "man_made=tower (0.40)", block)
        self.assertIn("names: Graves Light (0.80, both)", block)
        self.assertIn("kind: fixed_structure, extent: point_like", block)
        self.assertIn('description: "white conical tower on a rock"', block)
        self.assertIn("features: red lantern room", block)
        self.assertIn('unresolved: "possibly a monument instead"', block)

    def test_orphaned_audit_is_skipped(self):
        queries = ml.query_bundles({}, {9: audit()})
        self.assertEqual(queries, {})


class SignatureTest(unittest.TestCase):
    def test_identical_bundles_collapse_and_expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            feather = write_feather(Path(tmp) / "cat.feather")
            table = ml.build_map_signatures(feather)
            self.assertEqual(
                set(table),
                {"man_made=lighthouse; name=Graves Light",
                 "man_made=tower",
                 "object_class=LIGHTS"})
            self.assertEqual(table["man_made=lighthouse; name=Graves Light"],
                             ["osm:node:101"])
            self.assertEqual(sorted(table["man_made=tower"]),
                             ["osm:node:102", "osm:node:103"])
            # ENC row keeps its source prefix; the untagged row is gone.
            self.assertEqual(table["object_class=LIGHTS"], ["enc:lights:7"])


class ConfidenceMathTest(unittest.TestCase):
    def test_to_log_lr_is_clipped_log_odds(self):
        import math
        self.assertAlmostEqual(ml.to_log_lr(0.9), math.log(0.9 / 0.1))
        self.assertEqual(ml.to_log_lr(1.0), 4.0)   # clipped
        self.assertEqual(ml.to_log_lr(0.0), -4.0)  # clipped

    def test_global_no_match(self):
        matches = {"a": (0.7, "instance", "sig")}
        self.assertAlmostEqual(ml.global_no_match(matches, [0.9, 0.9]), 0.3)
        self.assertAlmostEqual(ml.global_no_match({}, [0.8, 0.6]), 0.7)
        self.assertEqual(ml.global_no_match({}, []), 1.0)

    def test_compatibility_table_round_trips_through_msgspec(self):
        table = ml.to_compatibility_table(
            "T7", {"osm:node:1": 2.0, "osm:node:2": -9.0},
            matcher_version="llm_chunked_v1_high", default_log_lr=-1.0)
        encoded = msgspec.json.encode([table])
        decoded = msgspec.json.decode(
            encoded, type=list[structs.CompatibilityTable])
        self.assertEqual(decoded[0].tracklet_id, "T7")
        self.assertEqual(decoded[0].entries[0].landmark_id, "osm:node:1")
        self.assertEqual(decoded[0].entries[0].log_lr, 2.0)
        # -9 clips to the table's clip_lo.
        self.assertEqual(decoded[0].entries[1].log_lr, -4.0)
        self.assertEqual(decoded[0].clip_hi, 4.0)


class EndToEndBuildAggregateTest(unittest.TestCase):
    """--build_only, fabricated results, --aggregate_only. No network."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.run_dir = tmp / "run"
        write_run(self.run_dir, split_ranges=True)
        self.feather = write_feather(tmp / "cat.feather")
        self.dataset_base = tmp / "datasets" / "tiny_harbor"
        self.dataset_base.mkdir(parents=True)
        self.match_dir = self.run_dir / "matching"
        self.flags = [
            "--run_dir", str(self.run_dir),
            "--dataset_base", str(self.dataset_base),
            "--feather", str(self.feather),
            "--query_batch", "2",
            "--chunk_size", "2",
            "--thinking_level", "HIGH",
            "--confidence_floor", "0.05",
            "--instance_max_rows", "1",
            "--model", "test-model",
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self):
        run_main(self.flags + ["--build_only"])

    def _fabricate_results(self):
        """One well-formed response per request: T1 matches the lighthouse
        signature (instance), T2 matches the two-row tower signature
        (instance, so aggregation must downgrade it), plus one below-floor
        match for T2 that must be dropped."""
        meta = json.loads((self.match_dir / "request_meta.json").read_text())
        sigs = sorted(ml.build_map_signatures(self.feather))
        chunks = [sigs[i:i + 2] for i in range(0, len(sigs), 2)]
        with open(self.match_dir / "results.jsonl", "w") as f:
            for key, record in meta.items():
                chunk = chunks[record["chunk_index"]]
                entries = []
                for i, tid in enumerate(record["batch_keys"]):
                    set_2 = []
                    if (tid == "T1" and
                            "man_made=lighthouse; name=Graves Light" in chunk):
                        set_2.append({
                            "set_2_id": chunk.index(
                                "man_made=lighthouse; name=Graves Light"),
                            "match_type": "instance", "confidence": 0.9})
                    if tid == "T2" and "man_made=tower" in chunk:
                        set_2.append({
                            "set_2_id": chunk.index("man_made=tower"),
                            "match_type": "instance", "confidence": 0.6})
                    if tid == "T2" and "object_class=LIGHTS" in chunk:
                        set_2.append({
                            "set_2_id": chunk.index("object_class=LIGHTS"),
                            "match_type": "category", "confidence": 0.01})
                    entries.append({"set_1_id": i, "set_2_matches": set_2,
                                    "no_match_confidence":
                                        0.2 if set_2 else 0.95,
                                    "uniqueness_score": 4})
                f.write(json.dumps({
                    "key": key,
                    "response": {"candidates": [{"content": {"parts": [
                        {"text": json.dumps({"matches": entries})}]}}]},
                }) + "\n")

    def test_build_then_aggregate(self):
        self._build()

        # --- request construction -------------------------------------------
        requests = [json.loads(line) for line in
                    (self.match_dir / "requests.jsonl").read_text()
                    .splitlines()]
        # 2 tracklets in one batch of 2, 3 signatures in 2 chunks.
        self.assertEqual(len(requests), 2)
        for record in requests:
            self.assertEqual(
                record["request"]["systemInstruction"]["parts"][0]["text"],
                ml.SYSTEM_PROMPT)
            self.assertEqual(
                record["request"]["generationConfig"]["responseSchema"],
                ml.SCHEMA)
        body = requests[0]["request"]["contents"][0]["parts"][0]["text"]
        self.assertIn("Set 1 (observed from the vessel):", body)
        self.assertIn("Set 2 (map database, arbitrary slice):", body)
        self.assertIn("Graves Light", body)

        # --- signatures.json -------------------------------------------------
        signatures = json.loads(
            (self.match_dir / "signatures.json").read_text())
        self.assertEqual(sorted(signatures["man_made=tower"]),
                         ["osm:node:102", "osm:node:103"])

        # --- settings.json: the provenance reference -------------------------
        settings = json.loads((self.match_dir / "settings.json").read_text())
        self.assertEqual(settings["model"], "test-model")
        self.assertEqual(settings["feather"], str(self.feather))
        self.assertEqual(settings["query_batch"], 2)
        self.assertEqual(settings["chunk_size"], 2)
        self.assertEqual(settings["confidence_floor"], 0.05)
        self.assertEqual(settings["instance_max_rows"], 1)
        self.assertEqual(settings["thinking_level"], "HIGH")
        self.assertEqual(
            settings["system_prompt_sha256"],
            hashlib.sha256(ml.SYSTEM_PROMPT.encode()).hexdigest())
        self.assertIn("git_commit", settings)
        self.assertEqual(settings["n_set1"], 2)
        self.assertEqual(settings["n_signatures"], 3)
        # The audit IS the support gate; the old flag must be gone.
        self.assertNotIn("min_supports", settings)
        self.assertIn("audit membership", settings["support_gate"])
        # The shared provenance manifest exists beside it.
        self.assertTrue((self.match_dir / "manifest.json").exists())

        # --- aggregate --------------------------------------------------------
        self._fabricate_results()
        run_main(self.flags + ["--aggregate_only"])

        matches = json.loads((self.match_dir / "matches.json").read_text())
        self.assertEqual(set(matches), {"T1", "T2"})

        t1 = matches["T1"]
        self.assertEqual(t1["n_landmarks"], 1)
        self.assertEqual(t1["matches"][0]["landmark_id"], "osm:node:101")
        self.assertEqual(t1["matches"][0]["match_type"], "instance")
        # Global null is 1 - best confidence, not a per-slice fusion.
        self.assertAlmostEqual(t1["no_match_confidence"], 0.1)
        self.assertEqual(t1["per_slice_no_match"]["n"], 2)

        t2 = matches["T2"]
        # The signature expanded to 2 rows > --instance_max_rows 1, so the
        # instance claim is downgraded in code rather than trusted; the
        # 0.01 match sits below the floor and is dropped entirely.
        self.assertEqual(t2["n_landmarks"], 2)
        self.assertEqual(t2["n_downgraded_to_category"], 2)
        for m in t2["matches"]:
            self.assertEqual(m["match_type"], "category")
            self.assertAlmostEqual(m["confidence"], 0.6)
        self.assertNotIn("enc:lights:7",
                         [m["landmark_id"] for m in t2["matches"]])

        # --- compatibility.json: decodes as the filter's structs -------------
        tables = msgspec.json.decode(
            (self.match_dir / "compatibility.json").read_bytes(),
            type=list[structs.CompatibilityTable])
        by_id = {t.tracklet_id: t for t in tables}
        self.assertEqual(set(by_id), {"T1", "T2"})

        import math
        t1_table = by_id["T1"]
        self.assertEqual(t1_table.matcher_version, "llm_chunked_v1_high")
        self.assertEqual(len(t1_table.entries), 1)
        self.assertAlmostEqual(t1_table.entries[0].log_lr,
                               math.log(0.9 / 0.1), places=6)
        # default = logit(max(1e-4, 1 - no_match) / n_signatures), verbatim.
        expected_default = ml.to_log_lr(max(1e-4, 1.0 - 0.1) / 3)
        self.assertAlmostEqual(t1_table.default_log_lr, expected_default,
                               places=6)
        self.assertEqual(t1_table.clip_lo, -4.0)
        self.assertEqual(t1_table.clip_hi, 4.0)
        self.assertEqual(t1_table.status, "fast")

        t2_table = by_id["T2"]
        self.assertEqual({e.landmark_id for e in t2_table.entries},
                         {"osm:node:102", "osm:node:103"})

    def test_missing_audit_is_a_clear_error(self):
        import shutil
        shutil.rmtree(self.run_dir / "semantic_audit")
        with self.assertRaises(SystemExit) as ctx:
            run_main(self.flags + ["--build_only"])
        self.assertIn("semantic audit", str(ctx.exception))

    def test_model_flag_is_required(self):
        flags = [f for f in self.flags if f not in ("--model", "test-model")]
        with self.assertRaises(SystemExit):
            run_main(flags + ["--build_only"])


if __name__ == "__main__":
    unittest.main()
