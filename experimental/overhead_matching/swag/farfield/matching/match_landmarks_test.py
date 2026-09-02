"""Tests for the matcher, on a tiny synthetic run. No network anywhere:
requests are built with --build_only, results are fabricated in the shape
the stage transport writes, and aggregation runs with --aggregate_only."""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import msgspec
import shapely

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import build_config
from experimental.overhead_matching.swag.farfield import llm_lifecycle
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.localization import structs
from experimental.overhead_matching.swag.farfield.matching import (
    match_landmarks as ml,
)

ANCHOR_LAT, ANCHOR_LON = 42.35, -71.05
DATASET = "tiny_harbor"
TRACKS_VERSION = "tracks-v1"
AUDITS_VERSION = "audits-v1"
CATALOG_VERSION = "catalog-v1"
MATCHES_VERSION = "matches-v1"


def write_feather(path: Path) -> Path:
    """Five rows: one unique signature, one two-row signature, one ENC row,
    one row with no far-field tags at all."""
    frame = schema.build_frame(
        ids=["('node', 101)", "('node', 102)", "('node', 103)",
             "('lights', 7)", "('node', 999)"],
        geometries=[
            shapely.Polygon([
                (ANCHOR_LON + 0.0008, ANCHOR_LAT + 0.0008),
                (ANCHOR_LON + 0.0012, ANCHOR_LAT + 0.0008),
                (ANCHOR_LON + 0.0012, ANCHOR_LAT + 0.0012),
                (ANCHOR_LON + 0.0008, ANCHOR_LAT + 0.0012),
                (ANCHOR_LON + 0.0008, ANCHOR_LAT + 0.0008),
            ]),
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
        "end_keyframe": max(keyframes),
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
    kept = verdict != "drop"
    return {
        "verdict": verdict,
        "single_object": kept,
        "drop_reason": "none" if kept else "insufficient_evidence",
        "landmark_kind": "fixed_structure",
        "valid_segments": ([{"start_t": 0, "end_t": 2}]
                           if kept else []),
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
        "strike_votes": [],
        "secondary_objects": [],
        "confidence": "high",
    }


def write_bound_inputs(root: Path, build_identity: str):
    """Publish the three exact typed upstreams consumed by matching."""
    tracks_dir = (root / "artifacts" / paths_lib.OBJECT_TRACKS / DATASET /
                  TRACKS_VERSION)
    tracks_document = {
        "range": {"name": "seg0"},
        "tracks": [track(1, [0, 1, 2]), track(2, [1, 2, 3]),
                   track(3, [4, 5]), track(4, [4, 5])],
    }
    with artifact.ArtifactDirectoryBuilder(
            tracks_dir,
            kind=paths_lib.OBJECT_TRACKS,
            dataset=DATASET,
            version=TRACKS_VERSION,
            generator="test:tracks",
            git_commit="test",
            arguments=(),
            config={"build_identity": build_identity},
            declared_outputs=("tracks_seg0.json",)) as builder:
        artifact.atomic_write_json(
            builder.output_path("tracks_seg0.json"), tracks_document)
    tracks_ref = artifact.open_artifact(tracks_dir)
    tracks_path = tracks_dir / "tracks_seg0.json"
    tracks_by_id = {item["track_id"]: item
                    for item in tracks_document["tracks"]}

    audit_dir = (root / "artifacts" / paths_lib.SEMANTIC_AUDITS / DATASET /
                 AUDITS_VERSION)
    requests = {}
    result_lines = []
    for tid, payload in ((1, audit()),
                         (2, audit(name="Customs Tower")),
                         (3, audit(verdict="drop"))):
        key = f"T{tid}"
        requests[key] = {
            "track_id": tid,
            "birth_keyframe": tracks_by_id[tid]["birth_keyframe"],
            "range": "seg0",
            "source_track_sha256": audit_io.canonical_sha256(
                tracks_by_id[tid]),
        }
        result_lines.append(artifact.canonical_json_bytes({
            "key": key,
            "response": {"candidates": [{"content": {"parts": [
                {"text": json.dumps(payload)}]}}]},
        }) + b"\n")
    meta = {
        "schema": audit_io.META_SCHEMA,
        "source_tracks": {
            "artifact_id": audit_io.source_artifact_id(tracks_ref),
            "file": tracks_path.name,
            "sha256": artifact.sha256_file(tracks_path),
        },
        "requests": requests,
    }
    with artifact.ArtifactDirectoryBuilder(
            audit_dir,
            kind=paths_lib.SEMANTIC_AUDITS,
            dataset=DATASET,
            version=AUDITS_VERSION,
            generator="test:audits",
            git_commit="test",
            arguments=(),
            upstreams=(tracks_ref,),
            config={
                "build_identity": build_identity,
                "phase": "canonical_results",
                "coverage": "complete",
                "n_expected": len(requests),
                "n_successful": len(requests),
            },
            declared_outputs=("audit_meta.json", "results.jsonl")) as builder:
        artifact.atomic_write_json(builder.output_path("audit_meta.json"), meta)
        artifact.atomic_write_file(
            builder.output_path("results.jsonl"), b"".join(result_lines))

    catalog_dir = (root / "artifacts" / paths_lib.CATALOGS / DATASET /
                   CATALOG_VERSION)
    with artifact.ArtifactDirectoryBuilder(
            catalog_dir,
            kind=paths_lib.CATALOGS,
            dataset=DATASET,
            version=CATALOG_VERSION,
            generator="test:catalog",
            git_commit="test",
            arguments=(),
            config={
                "schema": schema.FULL_ARTIFACT_SCHEMA,
                "source_coverage": {
                    "schema": "farfield_catalog_source_coverage/v2",
                    "status": "passed",
                    "message": "all requested source area covered",
                    "details": [],
                },
            },
            declared_outputs=("catalog.feather",)) as builder:
        write_feather(builder.output_path("catalog.feather"))
    return tracks_dir, audit_dir, catalog_dir


def write_build_config(root: Path, dataset_base: Path) -> tuple[Path, str]:
    config = {
        "artifacts": {
            "object_tracks_version": TRACKS_VERSION,
            "semantic_audits_version": AUDITS_VERSION,
            "catalogs_version": CATALOG_VERSION,
            "landmark_matches_version": MATCHES_VERSION,
        },
        "matching": {
            "model": "test-model",
            "query_batch": 2,
            "chunk_size": 2,
            "thinking_level": "HIGH",
            "confidence_floor": 0.05,
            "instance_max_rows": 1,
        },
        "execution": {
            "llm_transport": "on_demand",
            "batch_gcs_prefix": None,
            "approve_cost": False,
        },
        "cost": {"limit_usd": 5.0},
    }
    build_dir = root / "build"
    path = build_config.create(
        build_dir,
        dataset=DATASET,
        config=config,
        generator="test:build",
        inputs={"dataset_base": dataset_base, "farfield_root": root})
    document = build_config.load(build_dir)
    selected = {key: build_config.value(document, key)
                for key in ml.MATCHING_CONFIG_KEYS}
    return path, artifact.sha256_json(selected)


def run_main(argv):
    old = sys.argv
    sys.argv = ["match_landmarks"] + argv
    try:
        ml.main()
    finally:
        sys.argv = old


class QueryBundlesTest(unittest.TestCase):
    def test_one_entry_per_audited_track_drop_excluded(self):
        tracks = {
            1: track(1, [0, 1, 2]),
            2: track(2, [1, 2, 3]),
            3: track(3, [4, 5]),
            4: track(4, [4, 5]),
        }
        audits = {
            1: audit(),
            2: audit(name="Customs Tower"),
            3: audit(verdict="drop"),
        }
        queries = ml.query_bundles(tracks, audits)
        # 1 and 2 audited keep; 3 audited drop; 4 never audited.
        self.assertEqual(
            {key.rsplit("#", 1)[-1] for key in queries}, {"T1", "T2"})
        self.assertTrue(all(key.startswith("sha256:") for key in queries))

    def test_query_block_carries_the_audit_uncertainty(self):
        block = ml.format_query(audit())
        self.assertIn("tags: man_made=lighthouse (0.90); "
                      "man_made=tower (0.40)", block)
        self.assertIn("names: Graves Light (0.80, both)", block)
        self.assertIn("kind: fixed_structure, extent: point_like", block)
        self.assertIn('description: "white conical tower on a rock"', block)
        self.assertIn("features: red lantern room", block)
        self.assertIn('unresolved: "possibly a monument instead"', block)

    def test_orphaned_audit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no source track"):
            ml.query_bundles({}, {9: audit()})


class SignatureTest(unittest.TestCase):
    def test_identical_bundles_collapse_and_expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            feather = write_feather(Path(tmp) / "cat.feather")
            table = ml.build_map_signatures(feather)
            lighthouse = ml.signature({
                "man_made": "lighthouse", "name": "Graves Light"})
            tower = ml.signature({"man_made": "tower"})
            lights = ml.signature({"object_class": "LIGHTS"})
            self.assertEqual(set(table), {lighthouse, tower, lights})
            self.assertTrue(all(key.startswith("sha256:") for key in table))
            self.assertEqual(table[lighthouse]["display_label"],
                             "man_made=lighthouse; name=Graves Light")
            self.assertEqual(table[lighthouse]["landmark_ids"],
                             ["osm:node:101"])
            self.assertEqual(sorted(table[tower]["landmark_ids"]),
                             ["osm:node:102", "osm:node:103"])
            # ENC row keeps its source prefix; the untagged row is gone.
            self.assertEqual(table[lights]["landmark_ids"], ["enc:lights:7"])

    def test_digest_identity_does_not_confuse_delimiter_text(self):
        # Both old display strings are "a=b; c=d"; canonical tag JSON differs.
        self.assertEqual(
            ml.signature_display({"a": "b; c=d"}),
            ml.signature_display({"a": "b", "c": "d"}))
        self.assertNotEqual(
            ml.signature({"a": "b; c=d"}),
            ml.signature({"a": "b", "c": "d"}))

    def test_repeated_global_catalog_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.feather"
            schema.build_frame(
                ids=["node:1", "osm:node:1"],
                geometries=[shapely.Point(0, 0), shapely.Point(0, 0)],
                landmark_types=["osm", "osm"],
                tags=[{"man_made": "tower"}, {"man_made": "tower"}],
            ).to_feather(path)
            with self.assertRaisesRegex(ValueError, "globally namespaced"):
                ml.build_map_signatures(path)

    def test_building_inherits_smallest_named_complex(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "complex.feather"
            frame = schema.build_frame(
                ids=["way:1", "way:2", "way:3", "way:4"],
                geometries=[
                    shapely.box(0.0, 0.0, 10.0, 10.0),
                    shapely.box(2.0, 2.0, 8.0, 8.0),
                    shapely.box(3.0, 3.0, 4.0, 4.0),
                    shapely.box(20.0, 20.0, 21.0, 21.0),
                ],
                landmark_types=["osm"] * 4,
                tags=[
                    {"landuse": "residential", "name": "Outer Complex"},
                    {"landuse": "residential", "name": "Sejan Berche"},
                    {"building": "apartments", "name": "101"},
                    {"building": "apartments", "name": "101"},
                ],
            )
            frame.to_feather(path)
            table = ml.build_map_signatures(path)

        inherited = ml.signature({
            "building": "apartments",
            "complex:name": "Sejan Berche",
            "name": "101",
        })
        bare = ml.signature({"building": "apartments", "name": "101"})
        self.assertEqual(table[inherited]["landmark_ids"], ["osm:way:3"])
        self.assertEqual(table[bare]["landmark_ids"], ["osm:way:4"])


class ConfidenceMathTest(unittest.TestCase):
    def test_to_log_lr_is_clipped_log_odds(self):
        import math
        self.assertAlmostEqual(ml.to_log_lr(0.9), math.log(0.9 / 0.1))
        self.assertEqual(ml.to_log_lr(1.0), 4.0)   # clipped
        self.assertEqual(ml.to_log_lr(0.0), -4.0)  # clipped

    def test_global_no_match(self):
        matches = {"a": {"aggregate_confidence": 0.7}}
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


class ResponseValidationTest(unittest.TestCase):

    metadata = {
        "batch_keys": ["T1", "T2"],
        "chunk_index": 0,
        "chunk_signature_ids": ["sig-lighthouse", "sig-tower"],
    }

    @staticmethod
    def response(entries):
        return {"candidates": [{"content": {"parts": [
            {"text": json.dumps({"matches": entries})}]}}]}

    @staticmethod
    def entry(set_1_id):
        return {
            "set_1_id": set_1_id,
            "set_2_matches": [],
            "no_match_confidence": 0.95,
            "uniqueness_score": 3,
        }

    def test_no_match_is_valid_when_every_set1_item_is_present(self):
        result = ml.validate_matching_response(
            "q", self.response([self.entry(1), self.entry(0)]), self.metadata)
        self.assertEqual([item["set_1_id"] for item in result["matches"]],
                         [0, 1])

    def test_provider_thought_signature_metadata_is_accepted(self):
        response = self.response([self.entry(0), self.entry(1)])
        response["candidates"][0]["content"]["parts"][0][
            "thoughtSignature"] = "opaque-provider-signature"
        result = ml.validate_matching_response("q", response, self.metadata)
        self.assertEqual([item["set_1_id"] for item in result["matches"]],
                         [0, 1])

    def test_other_part_metadata_and_invalid_thought_signature_are_rejected(self):
        response = self.response([self.entry(0), self.entry(1)])
        response["candidates"][0]["content"]["parts"][0]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "optional provider"):
            ml.validate_matching_response("q", response, self.metadata)

        for signature in ("", None, 7):
            with self.subTest(signature=signature):
                response = self.response([self.entry(0), self.entry(1)])
                response["candidates"][0]["content"]["parts"][0][
                    "thoughtSignature"] = signature
                with self.assertRaisesRegex(ValueError, "nonempty string"):
                    ml.validate_matching_response("q", response, self.metadata)

    def test_missing_set1_item_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            ml.validate_matching_response(
                "q", self.response([self.entry(0)]), self.metadata)

    def test_duplicate_set1_item_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate set_1_id"):
            ml.validate_matching_response(
                "q", self.response([self.entry(0), self.entry(0)]),
                self.metadata)

    def test_unknown_fields_and_invalid_probabilities_are_rejected(self):
        entries = [self.entry(0), self.entry(1)]
        entries[0]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            ml.validate_matching_response(
                "q", self.response(entries), self.metadata)
        entries = [self.entry(0), self.entry(1)]
        entries[0]["no_match_confidence"] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            ml.validate_matching_response(
                "q", self.response(entries), self.metadata)


class EndToEndBuildAggregateTest(unittest.TestCase):
    """--build_only, fabricated results, --aggregate_only. No network."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.dataset_base = tmp / "datasets" / "tiny_harbor"
        self.dataset_base.mkdir(parents=True)
        self.build_config, orchestration_digest = write_build_config(
            tmp, self.dataset_base)
        document = build_config.load(self.build_config.parent)
        (self.tracks_dir, self.audit_dir,
         self.catalog_dir) = write_bound_inputs(
             tmp, document["build_identity"])
        self.feather = self.catalog_dir / "catalog.feather"
        self.match_dir = tmp / "landmark_matches" / MATCHES_VERSION
        self.work_dir = ml.matching_work_dir(self.match_dir)
        self.flags = [
            "--dataset", DATASET,
            "--dataset_base", str(self.dataset_base),
            "--tracks_dir", str(self.tracks_dir),
            "--audit_dir", str(self.audit_dir),
            "--catalog_dir", str(self.catalog_dir),
            "--output_dir", str(self.match_dir),
            "--build_config", str(self.build_config),
            "--orchestration_config_digest", orchestration_digest,
            "--online",
        ]
        self.aggregate_flags = [
            "--dataset", DATASET,
            "--output_dir", str(self.match_dir),
            "--aggregate_only",
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
        snapshot = llm_lifecycle.load_request_set(
            self.work_dir / llm_lifecycle.REQUEST_SET_NAME)
        semantic = json.loads(
            (self.work_dir / ml.WORK_SNAPSHOT_NAME).read_text())
        by_display = {
            entry["display_label"]: signature_id
            for signature_id, entry in semantic["signatures"].items()
        }
        lighthouse = by_display["man_made=lighthouse; name=Graves Light"]
        tower = by_display["man_made=tower"]
        lights = by_display["object_class=LIGHTS"]
        with open(self.work_dir / ml.TRANSPORT_RESULTS_NAME, "w") as f:
            for unit in snapshot.units:
                key = unit.key
                record = unit.metadata
                chunk = record["chunk_signature_ids"]
                entries = []
                for i, tid in enumerate(record["batch_keys"]):
                    local_id = tid.rsplit("#", 1)[-1]
                    set_2 = []
                    if (local_id == "T1" and
                            lighthouse in chunk):
                        set_2.append({
                            "set_2_id": chunk.index(lighthouse),
                            "match_type": "instance", "confidence": 0.9})
                    if local_id == "T2" and tower in chunk:
                        set_2.append({
                            "set_2_id": chunk.index(tower),
                            "match_type": "instance", "confidence": 0.6})
                    if local_id == "T2" and lights in chunk:
                        set_2.append({
                            "set_2_id": chunk.index(lights),
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


    def test_missing_audit_is_a_clear_error(self):
        flags = list(self.flags)
        flags[flags.index("--audit_dir") + 1] = str(
            Path(self._tmp.name) / "missing-audit")
        with self.assertRaises(SystemExit) as ctx:
            run_main(flags + ["--build_only"])
        self.assertIn("input artifact", str(ctx.exception))

    def test_byte_identical_audit_copy_is_the_same_artifact(self):
        alias = Path(self._tmp.name) / "alternate-audits"
        shutil.copytree(self.audit_dir, alias)
        flags = list(self.flags)
        flags[flags.index("--audit_dir") + 1] = str(alias)
        run_main(flags + ["--build_only"])

    def test_byte_identical_catalog_copy_is_the_same_artifact(self):
        alias = Path(self._tmp.name) / "alternate-catalog"
        shutil.copytree(self.catalog_dir, alias)
        flags = list(self.flags)
        flags[flags.index("--catalog_dir") + 1] = str(alias)
        run_main(flags + ["--build_only"])


    def test_corrupt_input_manifest_is_normalized_to_stage_error(self):
        (self.audit_dir / artifact.MANIFEST_NAME).write_text("not-json\n")
        with self.assertRaisesRegex(
                SystemExit, "invalid matching input artifact"):
            self._build()

    def test_catalog_without_passed_source_coverage_is_rejected(self):
        manifest_path = self.catalog_dir / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["config"]["source_coverage"]["status"] = "failed"
        artifact.atomic_write_json(manifest_path, document)
        with self.assertRaisesRegex(SystemExit, "status='passed'"):
            self._build()

    def test_one_missing_request_refuses_all_publication(self):
        self._build()
        self._fabricate_results()
        transport = self.work_dir / ml.TRANSPORT_RESULTS_NAME
        lines = transport.read_text().splitlines()
        self.assertGreater(len(lines), 1)
        transport.write_text(lines[0] + "\n")
        with self.assertRaisesRegex(
                llm_lifecycle.IncompleteCoverageError,
                "complete, unique successful coverage"):
            run_main(self.aggregate_flags)
        self.assertFalse(self.match_dir.exists())

    def test_duplicate_success_refuses_ambiguous_selection(self):
        self._build()
        self._fabricate_results()
        transport = self.work_dir / ml.TRANSPORT_RESULTS_NAME
        lines = transport.read_text().splitlines()
        transport.write_text("\n".join(lines + [lines[0]]) + "\n")
        with self.assertRaisesRegex(
                llm_lifecycle.IncompleteCoverageError,
                "duplicate valid responses"):
            run_main(self.aggregate_flags)
        self.assertFalse(self.match_dir.exists())

    def test_model_cannot_override_build_config(self):
        with self.assertRaises(SystemExit):
            run_main(self.flags + ["--model", "other", "--build_only"])

    def test_nonpositive_execution_controls_are_rejected(self):
        for flag in ("--parallel", "--poll_interval"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                run_main(self.flags + [flag, "0", "--build_only"])

    def test_aggregate_does_not_reopen_mutable_semantic_inputs(self):
        self._build()
        self._fabricate_results()
        self.tracks_dir.rename(self.tracks_dir.with_name("tracks-moved"))
        self.audit_dir.rename(self.audit_dir.with_name("audits-moved"))
        self.catalog_dir.rename(self.catalog_dir.with_name("catalog-moved"))
        self.build_config.rename(self.build_config.with_name("recipe-moved"))
        run_main(self.aggregate_flags)
        self.assertTrue((self.match_dir / ml.MATCHES_NAME).is_file())

    def test_aggregate_rejects_mutable_replacement_inputs(self):
        self._build()
        with self.assertRaises(SystemExit):
            run_main(self.aggregate_flags + [
                "--catalog_dir", str(self.catalog_dir)])

    def test_aggregate_rejects_snapshot_semantics_not_bound_to_requests(self):
        self._build()
        self._fabricate_results()
        snapshot_path = self.work_dir / ml.WORK_SNAPSHOT_NAME
        snapshot = json.loads(snapshot_path.read_text())
        first = sorted(snapshot["queries"])[0]
        snapshot["queries"][first] += "\nforged semantic replacement"
        snapshot_path.write_text(
            artifact.canonical_json_bytes(snapshot).decode() + "\n")
        with self.assertRaisesRegex(
                SystemExit, "do not exactly encode the frozen queries"):
            run_main(self.aggregate_flags)
        self.assertFalse(self.match_dir.exists())


if __name__ == "__main__":
    unittest.main()
