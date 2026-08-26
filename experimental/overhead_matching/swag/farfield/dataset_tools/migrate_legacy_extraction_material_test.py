"""The migration must make these readable without claiming they are current."""

import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    migrate_legacy_extraction_material as legacy,
)

LEGACY_MANIFEST = {
    "kind": "frame_landmarks",
    "dataset": "ds",
    "version": "v4",
    "generator": "//experimental/.../scripts:extract_gemini_landmarks",
    "git_commit": "cfe9b61d7384689bd7fec8fef31012ad6b5383c8",
    "created": "2026-08-19",
    "inputs": ["datasets/ds", "artifacts/pinhole_images/ds/v1"],
    "config": {"prompt": "osm_tags_farfield"},
    "notes": "Consumers glob sentences/results/**/predictions.jsonl",
}


def make_legacy(root: Path, version="v4", manifest=None) -> Path:
    target = root / "artifacts" / paths_lib.FRAME_LANDMARKS / "ds" / version
    (target / "sentences" / "results").mkdir(parents=True)
    (target / "sentences" / "results" / "predictions.jsonl").write_text("{}\n")
    (target / "sentence_requests").mkdir()
    (target / "sentence_requests" / "req.json").write_text("{}")
    (target / "gcs_prefix.txt").write_text("gs://x")
    document = dict(LEGACY_MANIFEST if manifest is None else manifest)
    document["version"] = version
    (target / artifact.MANIFEST_NAME).write_text(json.dumps(document))
    return target


class PlanTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_legacy_directory_is_recognised(self):
        make_legacy(self.root)
        plan = legacy.build_plan(self.root)
        self.assertEqual(plan["by_status"], {"migrate": 1})
        record = plan["directories"][0]
        self.assertEqual(record["from_kind"], "frame_landmarks")
        self.assertEqual(record["to_kind"], legacy.LEGACY_KIND)

    def test_a_current_artifact_is_not_touched(self):
        target = self.root / "artifacts" / paths_lib.FRAME_LANDMARKS / "ds" / "v9"
        with artifact.ArtifactDirectoryBuilder(
                target, kind=paths_lib.FRAME_LANDMARKS, dataset="ds",
                version="v9", generator="t", arguments=(), upstreams=(),
                config={}, declared_outputs=("predictions.jsonl",)) as builder:
            builder.output_path("predictions.jsonl").write_text("{}\n")
        self.assertEqual(legacy.build_plan(self.root)["by_status"], {})

    def test_an_unrecognised_layout_is_left_alone(self):
        target = self.root / "artifacts" / paths_lib.FRAME_LANDMARKS / "ds" / "v4"
        target.mkdir(parents=True)
        (target / artifact.MANIFEST_NAME).write_text(
            json.dumps(LEGACY_MANIFEST))
        self.assertEqual(legacy.build_plan(self.root)["by_status"],
                         {"unrecognized_layout": 1})


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.target = make_legacy(self.root)
        legacy.apply_plan(self.root, legacy.build_plan(self.root))

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_directory_is_now_readable(self):
        manifest = artifact.load_manifest(self.target)
        self.assertEqual(manifest.kind, legacy.LEGACY_KIND)
        self.assertEqual(len(manifest.content_digest), 64)

    def test_it_is_no_longer_addressable_as_frame_landmarks(self):
        """Re-kinding is the point. A current-schema manifest still saying
        frame_landmarks would let `open_artifact(expected_kind=...)` succeed
        on a directory holding no predictions.jsonl -- a silent wrong answer
        in place of today's loud parse error."""
        with self.assertRaises(artifact.ArtifactError):
            artifact.open_artifact(
                self.target, expected_kind=paths_lib.FRAME_LANDMARKS)

    def test_the_legacy_kind_is_not_a_pipeline_input(self):
        self.assertNotIn(legacy.LEGACY_KIND, paths_lib.ARTIFACT_KINDS)

    def test_the_original_manifest_is_kept_verbatim(self):
        """It is the only record of what this directory claimed about itself."""
        kept = json.loads(
            (self.target / "manifest.pre_contract.json").read_text())
        self.assertEqual(kept["kind"], "frame_landmarks")
        self.assertEqual(kept["notes"], LEGACY_MANIFEST["notes"])

    def test_the_payload_is_untouched(self):
        self.assertEqual(
            (self.target / "sentences" / "results"
             / "predictions.jsonl").read_text(), "{}\n")
        self.assertTrue((self.target / "sentence_requests" / "req.json").is_file())

    def test_the_producing_commit_is_preserved_not_invented(self):
        manifest = artifact.load_manifest(self.target)
        self.assertEqual(manifest.git_commit, LEGACY_MANIFEST["git_commit"])
        self.assertEqual(manifest.code_provenance["commit"],
                         LEGACY_MANIFEST["git_commit"])
        self.assertIn("no working diff was recorded",
                      manifest.code_provenance["note"])

    def test_a_migrated_directory_is_not_planned_again(self):
        """It now carries a schema, so it no longer reads as legacy. Running
        the tool twice cannot overwrite the kept original."""
        self.assertEqual(legacy.build_plan(self.root)["by_status"], {})
        self.assertTrue(
            (self.target / "manifest.pre_contract.json").is_file())


if __name__ == "__main__":
    unittest.main()
