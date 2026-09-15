import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.loci.publish_staged import (
    publish_staged,
)


class PublishStagedTest(unittest.TestCase):

    def test_publishes_only_declared_staged_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "kind" / "dataset" / "v1"
            staging = destination.with_name("v1.incomplete")
            staging.mkdir(parents=True)
            (staging / "result.txt").write_text("complete\n")

            reference = publish_staged(
                destination,
                generator="//test:generator",
                producer_command="generator --fixed-setting",
                declared_outputs=["result.txt"],
                upstream_paths=[],
                config={"setting": 1},
            )

            self.assertEqual(reference.kind, "kind")
            self.assertEqual(reference.dataset, "dataset")
            self.assertEqual(reference.version, "v1")
            self.assertFalse(staging.exists())
            manifest = artifact.load_manifest(destination)
            self.assertEqual(manifest.config, {"setting": 1})
            self.assertEqual(manifest.arguments, ("generator --fixed-setting",))

    def test_rejects_undeclared_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "kind" / "dataset" / "v1"
            staging = destination.with_name("v1.incomplete")
            staging.mkdir(parents=True)
            (staging / "result.txt").write_text("complete\n")
            (staging / "surprise.txt").write_text("unexpected\n")

            with self.assertRaises(artifact.ArtifactValidationError):
                publish_staged(
                    destination,
                    generator="//test:generator",
                    producer_command="generator",
                    declared_outputs=["result.txt"],
                    upstream_paths=[],
                    config={},
                )
            self.assertFalse(destination.exists())
            self.assertTrue(staging.exists())


if __name__ == "__main__":
    unittest.main()
