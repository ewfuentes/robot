import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import publication


class PublishedArtifactTest(unittest.TestCase):

    def _publish(self, destination: Path, *, kind: str = "object_tracks",
                 dataset: str = "campus", version: str = "v1"):
        with publication.published_artifact(
                destination,
                kind=kind,
                dataset=dataset,
                version=version,
                generator="publication_test",
                declared_outputs=("payload.json",)) as builder:
            artifact.atomic_write_json(
                builder.output_path("payload.json"), {"value": 1})
        self.assertIsNotNone(builder.artifact_ref)
        return builder.artifact_ref

    def test_canonical_artifact_refreshes_owning_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "artifacts/object_tracks/campus/v1"
            with mock.patch.object(publication.indexes, "refresh") as refresh:
                reference = self._publish(destination)

            refresh.assert_called_once_with(root.resolve())
            self.assertEqual(
                artifact.open_artifact(destination), reference)

    def test_canonical_run_refreshes_owning_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "runs/experiment_1/run_1"
            with mock.patch.object(publication.indexes, "refresh") as refresh:
                self._publish(
                    destination,
                    kind="localization_run",
                    dataset="campus",
                    version="run_1",
                )

            refresh.assert_called_once_with(root.resolve())

    def test_internal_work_artifact_does_not_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "work/requests/v1"
            with mock.patch.object(publication.indexes, "refresh") as refresh:
                self._publish(destination)

            refresh.assert_not_called()

    def test_run_shaped_non_run_artifact_does_not_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "runs/experiment_1/v1"
            with mock.patch.object(publication.indexes, "refresh") as refresh:
                self._publish(destination)

            refresh.assert_not_called()

    def test_refresh_failure_keeps_successful_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "artifacts/object_tracks/campus/v1"
            stderr = io.StringIO()
            with mock.patch.object(
                    publication.indexes, "refresh",
                    side_effect=RuntimeError("navigation unavailable")):
                with contextlib.redirect_stderr(stderr):
                    reference = self._publish(destination)

            self.assertIn("artifact published successfully", stderr.getvalue())
            self.assertIn("navigation unavailable", stderr.getvalue())
            self.assertEqual(artifact.open_artifact(destination), reference)

    def test_body_failure_neither_publishes_nor_refreshes(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = (Path(temporary)
                           / "artifacts/object_tracks/campus/v1")
            with mock.patch.object(publication.indexes, "refresh") as refresh:
                with self.assertRaisesRegex(RuntimeError, "producer failed"):
                    with publication.published_artifact(
                            destination,
                            kind="object_tracks",
                            dataset="campus",
                            version="v1",
                            generator="publication_test",
                            declared_outputs=("payload.json",)):
                        raise RuntimeError("producer failed")

            self.assertFalse(destination.exists())
            refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
