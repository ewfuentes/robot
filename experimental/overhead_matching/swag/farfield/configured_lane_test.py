import json
import shutil
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    configured_lane as subject,
)

KIND = "semantic_audits"
DATASET = "leg1"
VERSION = "v1"


def _publish(root: Path, *, dataset: str = DATASET,
             version: str = VERSION) -> artifact.ArtifactRef:
    lane = root / "artifacts" / KIND / dataset / version
    with artifact.ArtifactDirectoryBuilder(
            destination=lane, kind=KIND, dataset=dataset, version=version,
            generator="configured_lane_test", arguments={}, config={},
            declared_outputs=("audits.json",), upstreams=()) as builder:
        (builder.staging_dir / "audits.json").write_text(
            json.dumps({"audits": []}), encoding="utf-8")
    return builder.artifact_ref


def _document(root: Path, *, dataset: str = DATASET,
              version: str = VERSION) -> dict:
    return {
        "dataset": dataset,
        "inputs": {"farfield_root": str(root)},
        "config": {"artifacts": {f"{KIND}_version": version}},
    }


class ConfiguredLaneTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.ref = _publish(self.root)
        self.document = _document(self.root)
        self.lane = Path(self.ref.path)

    def test_the_configured_lane_itself_is_accepted(self):
        manifest = subject.require(
            self.ref, document=self.document, kind=KIND)
        self.assertEqual(manifest.kind, KIND)

    def test_a_byte_identical_copy_elsewhere_is_the_same_artifact(self):
        """The path is not the artifact. Same digests, same artifact."""
        alias = self.root / "scratch-copy"
        shutil.copytree(self.lane, alias)
        copied = artifact.ArtifactRef.from_dict(
            {**self.ref.to_dict(), "path": str(alias.resolve())})
        manifest = subject.require(copied, document=self.document, kind=KIND)
        self.assertEqual(manifest.kind, KIND)

    def _rebuild_the_lane_in_place(self):
        """Publishing is no-clobber, so a lane only changes if it is razed.

        That is the whole stale-input scenario: somebody rebuilt a version
        rather than choosing a new one, and refs recorded against the old
        contents now name bytes that no longer exist.
        """
        shutil.rmtree(self.lane)
        with artifact.ArtifactDirectoryBuilder(
                destination=self.lane, kind=KIND, dataset=DATASET,
                version=VERSION, generator="configured_lane_test",
                arguments={}, config={}, declared_outputs=("audits.json",),
                upstreams=()) as builder:
            (builder.staging_dir / "audits.json").write_text(
                json.dumps({"audits": [{"different": True}]}),
                encoding="utf-8")

    def test_a_ref_that_no_longer_matches_its_lane_is_rejected(self):
        """The one failure this check exists for: a stale input."""
        self._rebuild_the_lane_in_place()
        with self.assertRaisesRegex(subject.ConfiguredLaneError, "stale"):
            subject.require(self.ref, document=self.document, kind=KIND)

    def test_the_differing_fields_are_named(self):
        self._rebuild_the_lane_in_place()
        with self.assertRaisesRegex(
                subject.ConfiguredLaneError, "content_digest"):
            subject.require(self.ref, document=self.document, kind=KIND)

    def test_a_ref_for_another_dataset_is_rejected(self):
        """What the old path rule claimed to catch, caught by the manifest."""
        other = _publish(self.root, dataset="leg2")
        with self.assertRaisesRegex(
                subject.ConfiguredLaneError, "wrong semantic_audits ref"):
            subject.require(other, document=self.document, kind=KIND)

    def test_a_ref_for_another_version_is_rejected(self):
        other = _publish(self.root, version="v2")
        with self.assertRaisesRegex(
                subject.ConfiguredLaneError, "configured version"):
            subject.require(other, document=self.document, kind=KIND)

    def test_a_missing_lane_is_rejected_even_when_the_copy_is_valid(self):
        alias = self.root / "scratch-copy"
        shutil.copytree(self.lane, alias)
        shutil.rmtree(self.lane)
        copied = artifact.ArtifactRef.from_dict(
            {**self.ref.to_dict(), "path": str(alias.resolve())})
        with self.assertRaisesRegex(
                subject.ConfiguredLaneError, "not a valid artifact"):
            subject.require(copied, document=self.document, kind=KIND)

    def test_a_build_without_a_root_cannot_be_checked(self):
        document = _document(self.root)
        document["inputs"] = {}
        with self.assertRaisesRegex(
                subject.ConfiguredLaneError, "farfield_root"):
            subject.require(self.ref, document=document, kind=KIND)

    def test_a_build_without_a_configured_version_cannot_be_checked(self):
        document = _document(self.root)
        document["config"]["artifacts"] = {}
        with self.assertRaisesRegex(
                subject.ConfiguredLaneError, "exact semantic_audits lane"):
            subject.expected_lane(document, KIND)


if __name__ == "__main__":
    unittest.main()
