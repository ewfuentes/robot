import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    dataset as ds_lib,
    geometry as geo,
    testing,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    checksums,
    publish_mount_offset as pmo,
)


def sidecar(tmp: Path, **overrides) -> Path:
    record = {
        "mount_offset_deg": 215.0,
        "frame": geo.MOUNT_OFFSET_FRAME,
        "usable": True,
        "verdict": "AGREEING",
        "detail": "R=0.97",
        "generator": "farfield/calibration/sun_offset_check.py",
        "run": "r004",
        "git_commit": "cafebabe",
        "convention": geo.MOUNT_OFFSET_CONVENTION,
    }
    record.update(overrides)
    path = tmp / "sun_offset_check.json"
    path.write_text(json.dumps(record))
    return path


class LoadSidecarTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_accepts_a_usable_sidecar(self):
        record = pmo.load_sidecar(sidecar(self.dir))
        self.assertAlmostEqual(record["_offset_deg"], 215.0)

    def test_refuses_an_unusable_verdict(self):
        # The FIXED-OBJECT abstention: an angle its own estimator rejected
        # must never become dataset truth.
        path = sidecar(self.dir, usable=False, verdict="FIXED-OBJECT")
        with self.assertRaises(SystemExit) as ctx:
            pmo.load_sidecar(path)
        self.assertIn("FIXED-OBJECT", str(ctx.exception))

    def test_refuses_a_foreign_frame(self):
        path = sidecar(self.dir, frame="column_0")
        with self.assertRaises(SystemExit) as ctx:
            pmo.load_sidecar(path)
        self.assertIn("180", str(ctx.exception))

    def test_refuses_a_missing_angle(self):
        path = sidecar(self.dir, mount_offset_deg=None)
        with self.assertRaises(SystemExit):
            pmo.load_sidecar(path)

    def test_refuses_a_missing_file(self):
        with self.assertRaises(SystemExit):
            pmo.load_sidecar(self.dir / "nope.json")


class PublishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sidecar = sidecar(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def make_dataset(self, mount_offset="default"):
        meta = testing.default_metadata()
        if mount_offset == "none":
            del meta["mount_offset"]
        elif mount_offset != "default":
            meta["mount_offset"] = mount_offset
        base = testing.make_dataset(self.root / f"ds{id(mount_offset)}",
                                    n_frames=3, metadata=meta)
        return base

    def read_meta(self, base):
        return json.loads((base / "pipeline_metadata.json").read_text())

    def test_publishes_onto_a_dataset_with_no_offset(self):
        base = self.make_dataset(mount_offset="none")
        block = pmo.publish(base, self.sidecar, supersede_validated=False)
        self.assertAlmostEqual(block["mount_offset_deg"], 215.0)
        self.assertEqual(block["frame"], geo.MOUNT_OFFSET_FRAME)
        self.assertFalse(block["applied_to_heading_deg"])
        self.assertEqual(block["convention"], geo.MOUNT_OFFSET_CONVENTION)
        self.assertEqual(block["source"]["sidecar"], str(self.sidecar))
        self.assertTrue(block["published_by"]["git_commit"])
        self.assertNotIn("superseded", block)
        # The written block is consumable by the one validator.
        record = ds_lib.mount_offset_record(self.read_meta(base), base)
        self.assertAlmostEqual(record.offset_deg, 215.0)
        self.assertFalse(record.applied_to_heading_deg)

    def test_refuses_to_overwrite_an_accuracy_validated_block(self):
        # testing.default_metadata() ships 214.0 accuracy_validated.
        base = self.make_dataset()
        before = self.read_meta(base)["mount_offset"]
        with self.assertRaises(SystemExit) as ctx:
            pmo.publish(base, self.sidecar, supersede_validated=False)
        message = str(ctx.exception)
        self.assertIn("REFUSING", message)
        self.assertIn("accuracy_validated=true", message)
        self.assertIn("differ by 1.0 deg", message)
        # Untouched.
        self.assertEqual(self.read_meta(base)["mount_offset"], before)

    def test_supersede_preserves_the_old_block(self):
        base = self.make_dataset()
        old = self.read_meta(base)["mount_offset"]
        block = pmo.publish(base, self.sidecar, supersede_validated=True)
        self.assertEqual(block["superseded"], old)
        self.assertAlmostEqual(
            self.read_meta(base)["mount_offset"]["mount_offset_deg"], 215.0)

    def test_unvalidated_previous_block_is_replaced_and_kept(self):
        base = self.make_dataset(mount_offset={
            "mount_offset_deg": 180.0,
            "frame": geo.MOUNT_OFFSET_FRAME,
            "applied_to_heading_deg": False,
            "status": "manual",
            "accuracy_validated": False})
        block = pmo.publish(base, self.sidecar, supersede_validated=False)
        self.assertAlmostEqual(block["superseded"]["mount_offset_deg"], 180.0)

    def test_regenerates_checksums(self):
        base = self.make_dataset(mount_offset="none")
        target = base / checksums.CHECKSUM_FILE
        target.write_text("0000  ./pipeline_metadata.json\n")
        pmo.publish(base, self.sidecar, supersede_validated=False)
        lines = target.read_text().splitlines()
        # Every non-panorama file is covered, and the metadata's digest is
        # the file's real one (the old in-sweep writer left this stale).
        digests = {line.split("  ")[1]: line.split("  ")[0] for line in lines}
        self.assertIn("./pipeline_metadata.json", digests)
        self.assertEqual(
            digests["./pipeline_metadata.json"],
            checksums.file_sha256(base / "pipeline_metadata.json"))
        self.assertNotIn("0000", digests.values())

    def test_absent_checksum_file_is_not_invented(self):
        base = self.make_dataset(mount_offset="none")
        pmo.publish(base, self.sidecar, supersede_validated=False)
        self.assertFalse((base / checksums.CHECKSUM_FILE).exists())

    def test_not_a_dataset_directory(self):
        with self.assertRaises(SystemExit):
            pmo.publish(self.root / "empty", self.sidecar,
                        supersede_validated=False)


class ChecksumsTest(unittest.TestCase):
    def test_panoramas_and_derived_dirs_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = testing.make_dataset(Path(tmp) / "ds", n_frames=2)
            (base / checksums.CHECKSUM_FILE).write_text("")
            (base / "_manifests").mkdir()
            (base / "_manifests" / "vehicle_anchor.json").write_text("{}")
            n = checksums.regenerate(base)
            body = (base / checksums.CHECKSUM_FILE).read_text()
        self.assertGreater(n, 0)
        self.assertNotIn("panorama/", body)
        self.assertNotIn("_manifests", body)
        self.assertIn("./frames_gps.csv", body)


if __name__ == "__main__":
    unittest.main()
