import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.localization import sources


DATASET = "viewer_sources_test"


class SourceAncestryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def publish(self, name: str, kind: str, *, upstreams=()):
        path = self.root / name
        with artifact.ArtifactDirectoryBuilder(
                path, kind=kind, dataset=DATASET, version=name,
                generator="sources_test", git_commit="test", arguments=(),
                upstreams=upstreams, config={},
                declared_outputs=("payload.json",)) as builder:
            artifact.atomic_write_json(
                builder.output_path("payload.json"), {"name": name})
        return artifact.open_artifact(path)

    def test_run_must_descend_from_the_exact_tracker_and_audit_pair(self):
        tracks = self.publish("tracks-v1", paths_lib.OBJECT_TRACKS)
        audits = self.publish(
            "audits-v1", paths_lib.SEMANTIC_AUDITS, upstreams=(tracks,))
        matching = self.publish(
            "matching-v1", paths_lib.LANDMARK_MATCHES,
            upstreams=(tracks, audits))
        inputs = self.publish(
            "inputs-v1", paths_lib.LOCALIZATION_INPUTS,
            upstreams=(matching,))
        run = self.publish(
            "run-v1", sources.LOCALIZATION_RUN_KIND, upstreams=(inputs,))

        sources._validate_run_ancestry(  # noqa: SLF001
            run.path, tracks, audits)

        other_tracks = self.publish("tracks-v2", paths_lib.OBJECT_TRACKS)
        with self.assertRaisesRegex(
                sources.SourceContractError, "supplied object_tracks"):
            sources._validate_run_ancestry(  # noqa: SLF001
                run.path, other_tracks, audits)

        other_audits = self.publish(
            "audits-v2", paths_lib.SEMANTIC_AUDITS,
            upstreams=(other_tracks,))
        with self.assertRaisesRegex(
                sources.SourceContractError, "supplied semantic_audits"):
            sources._validate_run_ancestry(  # noqa: SLF001
                run.path, tracks, other_audits)


if __name__ == "__main__":
    unittest.main()
