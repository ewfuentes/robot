"""The survey must be read-only, and must never mistake a broken artifact for
an unattributed one -- those need opposite responses."""

import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    survey_artifact_identity as survey,
)

DIGEST = "a" * 64


def publish(root: Path, kind: str, dataset: str, version: str,
            config: dict) -> Path:
    target = root / "artifacts" / kind / dataset / version
    with artifact.ArtifactDirectoryBuilder(
            target, kind=kind, dataset=dataset, version=version,
            generator="survey_test", git_commit="c0ffee", arguments=(),
            upstreams=(), config=config,
            declared_outputs=("payload.txt",)) as builder:
        builder.output_path("payload.txt").write_text("x")
    return target


class SurveyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_an_artifact_without_an_identity_is_unattributed(self):
        publish(self.root, "object_tracks", "ds", "v1", {})
        report = survey.survey(self.root)
        self.assertEqual(report["by_state"], {"unattributed": 1})

    def test_an_artifact_with_one_is_attributed(self):
        publish(self.root, "object_tracks", "ds", "v1",
                {"artifact_identity": DIGEST})
        report = survey.survey(self.root)
        self.assertEqual(report["by_state"], {"attributed": 1})
        self.assertEqual(report["artifacts"][0]["identity"], DIGEST)

    def test_a_malformed_identity_is_not_reported_as_unattributed(self):
        """Opposite responses: an unattributed artifact can be adopted, a
        malformed one is a bug to look at. Collapsing them would quietly
        adopt corrupt provenance."""
        publish(self.root, "object_tracks", "ds", "v1",
                {"artifact_identity": "not-a-digest"})
        report = survey.survey(self.root)
        self.assertEqual(report["by_state"], {"malformed_identity": 1})

    def test_an_unreadable_manifest_is_reported_not_raised(self):
        target = self.root / "artifacts" / "object_tracks" / "ds" / "v1"
        target.mkdir(parents=True)
        (target / artifact.MANIFEST_NAME).write_text("{not json")
        report = survey.survey(self.root)
        self.assertEqual(report["by_state"], {"unreadable": 1})

    def test_runs_are_surveyed_alongside_artifact_lanes(self):
        run = self.root / "runs" / "260825_experiment" / "run_a"
        with artifact.ArtifactDirectoryBuilder(
                run, kind="localization_run", dataset="ds", version="run_a",
                generator="survey_test", git_commit="c0ffee", arguments=(),
                upstreams=(), config={},
                declared_outputs=("payload.txt",)) as builder:
            builder.output_path("payload.txt").write_text("x")
        report = survey.survey(self.root)
        self.assertEqual(report["n_artifacts"], 1)
        self.assertEqual(report["by_kind"], {"localization_run": 1})

    def test_the_survey_writes_nothing(self):
        target = publish(self.root, "object_tracks", "ds", "v1", {})
        before = {path: path.stat().st_mtime_ns
                  for path in sorted(self.root.rglob("*")) if path.is_file()}
        survey.survey(self.root)
        after = {path: path.stat().st_mtime_ns
                 for path in sorted(self.root.rglob("*")) if path.is_file()}
        self.assertEqual(before, after)
        self.assertTrue((target / "payload.txt").is_file())

    def test_a_side_output_is_not_reported_as_a_broken_artifact(self):
        """Two manifest formats share the filename. Feeding a viewer's
        provenance manifest to the artifact loader reported a healthy sidecar
        as corrupt -- found by running this survey against the real root."""
        viewer = self.root / "runs" / "260825_experiment" / "run_a.viewer"
        viewer.mkdir(parents=True)
        provenance.write(viewer, generator="viewer_test",
                         inputs={"run_dir": "/x"}, config={})
        report = survey.survey(self.root)
        self.assertEqual(report["by_state"], {"side_output": 1})

    def test_an_empty_root_is_not_an_error(self):
        report = survey.survey(self.root)
        self.assertEqual(report["n_artifacts"], 0)
        self.assertEqual(report["by_state"], {})

    def test_a_directory_without_a_manifest_is_skipped(self):
        stray = self.root / "artifacts" / "object_tracks" / "ds" / "notes"
        stray.mkdir(parents=True)
        (stray / "readme.txt").write_text("not an artifact")
        self.assertEqual(survey.survey(self.root)["n_artifacts"], 0)


if __name__ == "__main__":
    unittest.main()
