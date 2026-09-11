import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.matching import identity_review


DATASET = "review_test"


def write_matching(root: Path, version: str, landmark_id: str):
    destination = root / f"matching-{version}"
    matches = {
        "track@sha256:" + "1" * 64 + "#T1": {
            "matches": [{"landmark_id": landmark_id}],
        },
    }
    with artifact.ArtifactDirectoryBuilder(
            destination, kind=paths_lib.LANDMARK_MATCHES, dataset=DATASET,
            version=version, generator="test", git_commit="test",
            arguments=(), config={
                "phase": "canonical_results",
                "coverage": "complete",
                "n_expected": 1,
                "n_successful": 1,
                "n_tracklets_expected": 1,
                "n_tracklets_successful": 1,
            }, declared_outputs=("matches.json",)) as builder:
        artifact.atomic_write_json(
            builder.output_path("matches.json"), matches)
    return destination


class IdentityReviewTest(unittest.TestCase):
    def test_typed_publication_and_exact_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matching = write_matching(root, "m1", "osm:node:1")
            matching_ref, candidates = identity_review.matching_candidates(
                matching)
            draft = identity_review.draft_document(matching_ref, candidates)
            row = draft["rows"][0]
            row.update({
                "decision": "confirmed",
                "landmark_ids": ["osm:node:1"],
                "reviewer": "reviewer@example.com",
                "timestamp": "2026-08-24T12:00:00-04:00",
                "notes": "confirmed against the source imagery",
            })
            draft_path = root / "draft.json"
            draft_path.write_text(json.dumps(draft))
            output = root / "review"
            reference = identity_review.publish(
                dataset=DATASET, matching_dir=matching,
                input_json=draft_path, output_dir=output, version="r1")
            self.assertEqual(reference.kind,
                             identity_review.IDENTITY_REVIEW_KIND)
            loaded_ref, loaded = identity_review.load(
                output, expected_matching_ref=matching_ref)
            self.assertEqual(loaded_ref, reference)
            self.assertEqual(loaded.decisions[0].decision, "confirmed")
            self.assertEqual(loaded.decisions[0].landmark_ids,
                             ("osm:node:1",))

    def test_stale_matching_and_off_candidate_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = write_matching(root, "m1", "osm:node:1")
            substitute = write_matching(root, "m2", "osm:node:2")
            original_ref, candidates = identity_review.matching_candidates(
                original)
            draft = identity_review.draft_document(original_ref, candidates)
            row = draft["rows"][0]
            row.update({
                "decision": "confirmed",
                "landmark_ids": ["osm:node:1"],
                "reviewer": "reviewer",
                "timestamp": "2026-08-24T16:00:00Z",
                "notes": "",
            })
            draft_path = root / "draft.json"
            draft_path.write_text(json.dumps(draft))
            output = root / "review-stale"
            with self.assertRaisesRegex(
                    identity_review.IdentityReviewError, "stale"):
                identity_review.publish(
                    dataset=DATASET, matching_dir=substitute,
                    input_json=draft_path, output_dir=output, version="r1")
            self.assertFalse(output.exists())

            row["landmark_ids"] = ["osm:node:not-a-candidate"]
            draft_path.write_text(json.dumps(draft))
            with self.assertRaisesRegex(
                    identity_review.IdentityReviewError,
                    "outside the exact machine candidates"):
                identity_review.publish(
                    dataset=DATASET, matching_dir=original,
                    input_json=draft_path, output_dir=output, version="r1")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
