import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import provenance


class WriteReadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "artifact"

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_records_the_reproduction_surface(self):
        provenance.write(
            self.dir, generator="farfield.tracking.track",
            inputs={"dataset_base": "/data/x", "video": "/data/v.mp4"},
            config={"epoch_keyframes": 5}, notes="unit test")
        doc = provenance.read(self.dir)
        self.assertEqual(doc["schema"], provenance.SCHEMA)
        self.assertEqual(doc["generator"], "farfield.tracking.track")
        self.assertEqual(doc["inputs"]["video"], "/data/v.mp4")
        self.assertEqual(doc["config"], {"epoch_keyframes": 5})
        # argv, git_commit and created are always present, whatever their
        # values are in this environment.
        self.assertIsInstance(doc["argv"], list)
        self.assertTrue(doc["git_commit"])
        self.assertIn("T", doc["created"])

    def test_read_missing_manifest_is_a_pointed_error(self):
        self.dir.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError) as ctx:
            provenance.read(self.dir)
        self.assertIn("manifest", str(ctx.exception))

    def test_extra_fields_cannot_shadow_standard_ones(self):
        with self.assertRaises(ValueError):
            provenance.write(self.dir, generator="g", inputs={}, config={},
                             extra={"git_commit": "spoofed"})
        provenance.write(self.dir, generator="g", inputs={}, config={},
                         extra={"kind": "object_tracks"})
        self.assertEqual(provenance.read(self.dir)["kind"], "object_tracks")


class DigestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.parent = Path(self.tmp.name) / "catalogs" / "ds"
        self.v1 = self.parent / "v1"
        self.v1.mkdir(parents=True)
        (self.v1 / "table.feather").write_bytes(b"payload")

    def tearDown(self):
        self.tmp.cleanup()

    def test_digest_ignores_the_manifest_itself(self):
        before = provenance.digest_dir(self.v1)
        provenance.write(self.v1, generator="g", inputs={}, config={})
        self.assertEqual(provenance.digest_dir(self.v1), before)

    def test_digest_changes_with_content(self):
        before = provenance.digest_dir(self.v1)
        (self.v1 / "table.feather").write_bytes(b"different")
        self.assertNotEqual(provenance.digest_dir(self.v1), before)

    def test_byte_identical_new_version_is_refused(self):
        digest = provenance.digest_dir(self.v1)
        provenance.write(self.v1, generator="g", inputs={}, config={},
                         content_digest=digest)
        v2 = self.parent / "v2"
        v2.mkdir()
        (v2 / "table.feather").write_bytes(b"payload")
        with self.assertRaises(ValueError) as ctx:
            provenance.check_version_is_new(v2, provenance.digest_dir(v2))
        self.assertIn("byte-identical", str(ctx.exception))

    def test_genuinely_new_version_passes(self):
        digest = provenance.digest_dir(self.v1)
        provenance.write(self.v1, generator="g", inputs={}, config={},
                         content_digest=digest)
        v2 = self.parent / "v2"
        v2.mkdir()
        (v2 / "table.feather").write_bytes(b"new payload")
        provenance.check_version_is_new(v2, provenance.digest_dir(v2))


if __name__ == "__main__":
    unittest.main()
