import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    landmark_positive_set as positive_set,
)


DATASET = "test_dataset"
SIGNATURE_TAGS = {"man_made": "tower"}
SIGNATURE = positive_set.format_signature(SIGNATURE_TAGS)
SIGNATURE_ID = positive_set.signature_id(SIGNATURE_TAGS)
LANDMARK_ID = "osm:node:1"


def signature_entry(tags: dict, landmark_ids: list[str]) -> dict:
    return {
        "canonical_tags": tags,
        "display_label": positive_set.format_signature(tags),
        "landmark_ids": landmark_ids,
    }


def match_entry(landmark_id: str = LANDMARK_ID, confidence: float = 0.9) -> dict:
    return {
        "landmark_id": landmark_id,
        "per_call_candidate_scores": [confidence],
        "aggregate_confidence": confidence,
        "aggregation_rule": positive_set.CANDIDATE_AGGREGATION_RULE,
        "match_type": "instance",
        "signature_id": SIGNATURE_ID,
        "signature_display": SIGNATURE,
    }


def publish_simple_artifact(
        root: Path, kind: str, version: str, output_name: str,
) -> tuple[Path, artifact.ArtifactRef]:
    output_dir = root / kind / version
    with artifact.ArtifactDirectoryBuilder(
            output_dir,
            kind=kind,
            dataset=DATASET,
            version=version,
            generator="landmark_positive_set_test",
            git_commit="test",
            declared_outputs=(output_name,)) as builder:
        artifact.atomic_write_file(builder.output_path(output_name), b"{}\n")
    return output_dir, artifact.open_artifact(output_dir)


def publish_catalog(
        root: Path, version: str = "catalog_v1",
) -> tuple[Path, artifact.ArtifactRef]:
    return publish_simple_artifact(
        root, paths_lib.CATALOGS, version, positive_set.CATALOG_PAYLOAD)


def default_matches() -> dict:
    return {
        "LT1": {
            "n_landmarks": 1,
            "n_signatures": 1,
            "matches": [match_entry()],
        },
        "LT2": {
            "n_landmarks": 0,
            "n_signatures": 0,
            "matches": [],
        },
    }


def publish_matching(
        root: Path, catalog_ref: artifact.ArtifactRef, *,
        matches: dict | None = None,
        signatures: dict | None = None,
        config: dict | None = None,
        declared_outputs: tuple[str, ...] | None = None,
) -> tuple[Path, artifact.ArtifactRef]:
    _, tracks_ref = publish_simple_artifact(
        root, paths_lib.OBJECT_TRACKS, "tracks_v1", "tracks.json")
    _, audits_ref = publish_simple_artifact(
        root, paths_lib.SEMANTIC_AUDITS, "audits_v1", "audits.json")
    matches = default_matches() if matches is None else matches
    signatures = (
        {SIGNATURE_ID: signature_entry(SIGNATURE_TAGS, [LANDMARK_ID])}
        if signatures is None else signatures)
    manifest_config = {
        "phase": "canonical_results",
        "coverage": "complete",
        "n_expected": 3,
        "n_successful": 3,
        "n_tracklets_expected": len(matches),
        "n_tracklets_successful": len(matches),
    }
    if config:
        manifest_config.update(config)
    outputs = (positive_set.MATCHING_OUTPUTS
               if declared_outputs is None else declared_outputs)
    output_dir = root / paths_lib.LANDMARK_MATCHES / "matches_v1"
    with artifact.ArtifactDirectoryBuilder(
            output_dir,
            kind=paths_lib.LANDMARK_MATCHES,
            dataset=DATASET,
            version="matches_v1",
            generator="landmark_positive_set_test",
            git_commit="test",
            upstreams=(tracks_ref, audits_ref, catalog_ref),
            config=manifest_config,
            declared_outputs=outputs) as builder:
        for output in outputs:
            if output == positive_set.MATCHES_PAYLOAD:
                artifact.atomic_write_json(builder.output_path(output), matches)
            elif output == positive_set.SIGNATURES_PAYLOAD:
                artifact.atomic_write_json(
                    builder.output_path(output), signatures)
            else:
                artifact.atomic_write_file(
                    builder.output_path(output), b"{}\n")
    return output_dir, artifact.open_artifact(output_dir)


class PositiveSetMatchingContractTest(unittest.TestCase):

    def test_build_uses_final_matches_and_records_exact_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog_ref = publish_catalog(root)
            matching_dir, matching_ref = publish_matching(root, catalog_ref)
            record = positive_set.build(matching_dir)
        self.assertEqual(record["schema"], positive_set.POSITIVE_SET_SCHEMA)
        self.assertEqual(record["matching"], matching_ref.to_dict())
        self.assertEqual(record["catalog"], catalog_ref.to_dict())
        self.assertEqual(record["n_tracklets"], 2)
        self.assertEqual(record["n_positives"], 1)
        self.assertEqual(record["positives"], [{
            "tracklet_id": "LT1",
            "landmark_id": LANDMARK_ID,
            "signature_id": SIGNATURE_ID,
            "signature_display": SIGNATURE,
            "match_type": "instance",
            "aggregate_confidence": 0.9,
        }])
        self.assertNotIn("position_sigma_m", record)
        self.assertNotIn("pairing_dir", record)

    def test_main_writes_schema_v2_atomically_loadable_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog_ref = publish_catalog(root)
            matching_dir, matching_ref = publish_matching(root, catalog_ref)
            output = root / "guards" / "positive.json"
            written = positive_set.main(matching_dir, output)
            loaded, loaded_matching, loaded_catalog = (
                positive_set.load_positive_set(output))
        self.assertEqual(loaded, written)
        self.assertEqual(loaded_matching, matching_ref)
        self.assertEqual(loaded_catalog, catalog_ref)

    def test_main_refuses_to_clobber_an_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog_ref = publish_catalog(root)
            matching_dir, _ = publish_matching(root, catalog_ref)
            output = root / "positive.json"
            original = b"curated guard must remain unchanged\n"
            output.write_bytes(original)
            with self.assertRaises(FileExistsError):
                positive_set.main(matching_dir, output)
            self.assertEqual(output.read_bytes(), original)

    def test_requires_complete_request_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog_ref = publish_catalog(root)
            matching_dir, _ = publish_matching(
                root, catalog_ref, config={"n_successful": 2})
            with self.assertRaisesRegex(
                    positive_set.PositiveSetError, "successful result"):
                positive_set.build(matching_dir)

    def test_requires_complete_tracklet_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog_ref = publish_catalog(root)
            matching_dir, _ = publish_matching(
                root, catalog_ref, config={"n_tracklets_successful": 1})
            with self.assertRaisesRegex(
                    positive_set.PositiveSetError, "final record"):
                positive_set.build(matching_dir)

    def test_requires_canonical_phase_and_exact_declared_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog_ref = publish_catalog(root)
            matching_dir, _ = publish_matching(
                root, catalog_ref, config={"phase": "requests"})
            with self.assertRaisesRegex(
                    positive_set.PositiveSetError, "canonical_results"):
                positive_set.build(matching_dir)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog_ref = publish_catalog(root)
            outputs = tuple(
                item for item in positive_set.MATCHING_OUTPUTS
                if item != "compatibility.json")
            matching_dir, _ = publish_matching(
                root, catalog_ref, declared_outputs=outputs)
            with self.assertRaisesRegex(
                    positive_set.PositiveSetError, "canonical final outputs"):
                positive_set.build(matching_dir)

    def test_match_must_reference_signature_and_landmark_binding(self):
        cases = (
            ({
                positive_set.signature_id({"other": "signature"}):
                    signature_entry({"other": "signature"}, [LANDMARK_ID]),
            }, "unknown signature"),
            ({
                SIGNATURE_ID:
                    signature_entry(SIGNATURE_TAGS, ["osm:node:2"]),
            }, "not bound"),
        )
        for signatures, message in cases:
            with self.subTest(signatures=signatures):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    _, catalog_ref = publish_catalog(root)
                    matching_dir, _ = publish_matching(
                        root, catalog_ref, signatures=signatures)
                    with self.assertRaisesRegex(
                            positive_set.PositiveSetError, message):
                        positive_set.build(matching_dir)

    def test_match_must_preserve_canonical_aggregation_and_display(self):
        cases = (
            ({"signature_display": "not the canonical display"},
             "signature display"),
            ({"per_call_candidate_scores": [0.8, 0.7],
              "aggregate_confidence": 0.7},
             "not the maximum"),
            ({"aggregation_rule": "some_other_rule"},
             "aggregation rule"),
        )
        for updates, message in cases:
            with self.subTest(updates=updates):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    _, catalog_ref = publish_catalog(root)
                    matches = default_matches()
                    matches["LT1"]["matches"][0].update(updates)
                    matching_dir, _ = publish_matching(
                        root, catalog_ref, matches=matches)
                    with self.assertRaisesRegex(
                            positive_set.PositiveSetError, message):
                        positive_set.build(matching_dir)

    def test_signature_metadata_is_bound_to_its_digest_and_display(self):
        cases = (
            ({
                SIGNATURE_ID:
                    signature_entry({"man_made": "lighthouse"}, [LANDMARK_ID]),
            }, "digest"),
            ({
                SIGNATURE_ID: {
                    **signature_entry(SIGNATURE_TAGS, [LANDMARK_ID]),
                    "display_label": "not canonical",
                },
            }, "display label"),
        )
        for signatures, message in cases:
            with self.subTest(signatures=signatures):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    _, catalog_ref = publish_catalog(root)
                    matching_dir, _ = publish_matching(
                        root, catalog_ref, signatures=signatures)
                    with self.assertRaisesRegex(
                            positive_set.PositiveSetError, message):
                        positive_set.build(matching_dir)


    def test_manifest_tracklet_count_must_match_matches_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog_ref = publish_catalog(root)
            matching_dir, _ = publish_matching(
                root, catalog_ref, config={"n_tracklets_expected": 3,
                                           "n_tracklets_successful": 3})
            with self.assertRaisesRegex(
                    positive_set.PositiveSetError, "tracklet count"):
                positive_set.build(matching_dir)

    def test_expected_catalog_identity_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, catalog_ref = publish_catalog(root)
            _, other_ref = publish_catalog(root, "catalog_v2")
            matching_dir, _ = publish_matching(root, catalog_ref)
            with self.assertRaisesRegex(
                    positive_set.PositiveSetError, "different catalog"):
                positive_set.open_matching_artifact(
                    matching_dir, expected_catalog_ref=other_ref)


class PositiveSetSchemaTest(unittest.TestCase):

    def test_legacy_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy.json"
            source.write_text(json.dumps({
                "pairing_dir": "old",
                "position_sigma_m": 25.0,
                "positives": [],
            }))
            with self.assertRaisesRegex(
                    positive_set.PositiveSetError, "keys"):
                positive_set.load_positive_set(source)

    def test_display_collision_cannot_satisfy_digest_recall(self):
        one_tag = {"a": "x; b=y"}
        two_tags = {"a": "x", "b": "y"}
        self.assertEqual(
            positive_set.format_signature(one_tag),
            positive_set.format_signature(two_tags),
        )
        one_id = positive_set.signature_id(one_tag)
        two_id = positive_set.signature_id(two_tags)
        self.assertNotEqual(one_id, two_id)

        record = {"signature_id": one_id}
        score, lost = positive_set.recall(
            {"positives": [record]},
            {two_id},
        )
        self.assertEqual(score, 0.0)
        self.assertEqual(lost, [record])

    def test_loose_catalog_file_is_not_an_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            loose_dir = Path(tmp) / "loose"
            loose_dir.mkdir()
            (loose_dir / positive_set.CATALOG_PAYLOAD).write_bytes(b"loose")
            with self.assertRaises(artifact.ArtifactValidationError):
                positive_set.open_catalog_artifact(loose_dir)


if __name__ == "__main__":
    unittest.main()
