import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    geometry,
    nominal_forward,
)
from experimental.overhead_matching.swag.farfield.calibration import (
    nominal_forward_review as subject,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import checksums


class NominalForwardReviewTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "review_dataset"
        panorama = self.dataset / "frames"
        panorama.mkdir(parents=True)
        (self.dataset / "panorama").symlink_to("frames")
        metadata = {
            "dataset_name": self.dataset.name,
            "source": "fixture",
            "is_equirectangular": True,
            "north_aligned": False,
            "intrinsics_csv": "intrinsics.csv",
            "azimuth_convention": {
                "images_rotated": False,
                "camera_frame": geometry.CAMERA_FRAME,
                "frame": "camera (as captured)",
                "bearing_increases": "left_to_right",
            },
        }
        (self.dataset / "pipeline_metadata.json").write_text(
            json.dumps(metadata, sort_keys=True) + "\n")
        gps_rows = []
        intrinsics_rows = []
        for index in range(5):
            pano_id = f"f{index:04d}"
            latitude = 42.0 + index * 0.0001
            longitude = -71.0
            stem = f"{pano_id},{latitude:.7f},{longitude:.7f},"
            image = Image.new("RGB", (360, 180),
                              (30 + index * 20, 70, 120))
            image.save(panorama / f"{stem}.jpg", format="JPEG")
            gps_rows.append({
                "idx": index,
                "latitude": f"{latitude:.7f}",
                "longitude": f"{longitude:.7f}",
                "dist_m": index * 10,
                "video_t_s": index * 2,
            })
            intrinsics_rows.append({
                "idx": index,
                "pano_id": pano_id,
                "projection": "equirectangular",
                "width": 360,
                "height": 180,
                "computed_compass_angle_true_deg": "",
                "compass_angle_true_deg": "",
                "heading_optical_axis_true_deg": "",
                "heading_column0_true_deg": "",
                "selected_heading_source": "",
            })
        self._write_csv(
            self.dataset / "frames_gps.csv",
            ["idx", "latitude", "longitude", "dist_m", "video_t_s"],
            gps_rows)
        self._write_csv(
            self.dataset / "intrinsics.csv",
            ["idx", "pano_id", "projection", "width", "height",
             "computed_compass_angle_true_deg", "compass_angle_true_deg",
             "heading_optical_axis_true_deg", "heading_column0_true_deg",
             "selected_heading_source"],
            intrinsics_rows)
        (self.dataset / checksums.CHECKSUM_FILE).write_text("")
        checksums.regenerate(self.dataset)
        self.bundle_index = 0

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_csv(path, fields, rows):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields,
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def _review_manifest(self, evidence_frame_ids):
        self.bundle_index += 1
        output = self.root / f"review_{self.bundle_index}"
        subject.create_review_bundle(
            self.dataset, output,
            evidence_frame_ids=tuple(evidence_frame_ids),
            display_width=600)
        return output / "review_manifest.json"

    def test_default_bundle_is_review_only_and_deterministic(self):
        output = self.root / "review"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(subject.main([
                "--dataset_base", str(self.dataset),
                "--output_dir", str(output),
                "--evidence_count", "3",
                "--display_width", "600",
            ]), 0)
        manifest = json.loads((output / "review_manifest.json").read_text())
        template = json.loads((output / "record_template.json").read_text())
        self.assertEqual(manifest["schema"], subject.BUNDLE_SCHEMA)
        self.assertIs(manifest["calibration_authority"], False)
        self.assertIs(manifest["safety"]["contains_approved_nominal_forward"],
                      False)
        self.assertEqual(
            manifest["selection"], {
                "method": "deterministic_evenly_spaced_v1",
                "evidence_frame_ids": ["f0000", "f0002", "f0004"],
            })
        self.assertIs(template["calibration_authority"], False)
        self.assertIsNone(template["bearing_camera_cw_deg"])
        self.assertIsNone(template["panorama_column"])
        self.assertNotIn("approved", template)
        with Image.open(output / "contact_sheet.png") as contact_sheet:
            self.assertEqual(contact_sheet.width, 600)
            self.assertGreater(contact_sheet.height, 600)
        with self.assertRaises(FileExistsError):
            subject.create_review_bundle(
                self.dataset, output, evidence_count=1, display_width=600)

    def test_explicit_evidence_and_exact_grid_columns(self):
        validated = subject.validate_dataset(self.dataset)
        selected, method = subject.select_evidence(
            validated, evidence_frame_ids=("f0004", "f0001"))
        self.assertEqual(method, "explicit")
        self.assertEqual([item.pano_id for item in selected],
                         ["f0004", "f0001"])
        grid = {item["panorama_column"]: item["bearing_camera_cw_deg"]
                for item in subject.column_grid(360, 30.0)}
        self.assertEqual(grid[180], 0.0)
        self.assertEqual(grid[0], 180.0)
        for column, bearing in grid.items():
            self.assertAlmostEqual(
                bearing,
                float(geometry.azimuth_of_pano_column(column, 360)) % 360.0)

    def test_strict_contract_rejects_heading_authority_and_bad_geometry(self):
        path = self.dataset / "intrinsics.csv"
        rows = list(csv.DictReader(path.open()))
        rows[0]["selected_heading_source"] = "gps course candidate"
        self._write_csv(path, list(rows[0]), rows)
        with self.assertRaisesRegex(subject.ReviewError,
                                    "unapproved heading authority"):
            subject.validate_dataset(self.dataset)

        rows[0]["selected_heading_source"] = ""
        self._write_csv(path, list(rows[0]), rows)
        first = next((self.dataset / "frames").glob("*.jpg"))
        Image.new("RGB", (360, 179), "black").save(first, format="JPEG")
        with self.assertRaisesRegex(subject.ReviewError,
                                    "dimensions disagree|2:1"):
            subject.validate_dataset(self.dataset)

    def test_mapillary_orientation_diagnostics_are_not_calibration_authority(self):
        metadata_path = self.dataset / "pipeline_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["source"] = "mapillary"
        metadata_path.write_text(json.dumps(metadata) + "\n")
        path = self.dataset / "intrinsics.csv"
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.update({
                "computed_compass_angle_true_deg": "10.0",
                "compass_angle_true_deg": "12.0",
                "heading_optical_axis_true_deg": "10.0",
                "heading_column0_true_deg": "190.0",
                "selected_heading_source": "computed_compass_angle",
            })
        self._write_csv(path, list(rows[0]), rows)

        subject.validate_dataset(self.dataset)

    def test_finalize_from_bearing_is_explicit_valid_and_no_overwrite(self):
        output = self.dataset / subject.NOMINAL_FORWARD_NAME
        checksum_before = (self.dataset / checksums.CHECKSUM_FILE).read_bytes()
        review_manifest = self._review_manifest(("f0001", "f0003"))
        document = subject.finalize_nominal_forward(
            self.dataset, output,
            version="human-v1",
            mounting_id="rig-2026",
            review_manifest=review_manifest,
            bearing_camera_cw_deg=20.0,
            panorama_column=None,
            uncertainty_deg=1.5,
            operator="Human Reviewer",
            approved_at="2026-08-24T15:30:00-04:00",
            notes="Boat bow aligned in both evidence frames.",
            evidence_frame_ids=("f0001", "f0003"),
            approve_as_authority=True)
        self.assertIs(document["approved"], True)
        self.assertAlmostEqual(document["panorama_column"], 200.0)
        loaded = nominal_forward.load(
            output, expected_dataset=self.dataset.name)
        self.assertEqual(loaded.bearing_camera_cw_deg, 20.0)
        self.assertEqual(
            loaded.notes,
            "Boat bow aligned in both evidence frames.\n"
            f"{subject._REVIEW_DIGEST_PREFIX}"
            f"{subject._sha256(review_manifest)}")
        self.assertGreater(checksums.verify(self.dataset), 0)
        checksum_after = (self.dataset / checksums.CHECKSUM_FILE).read_bytes()
        self.assertNotEqual(checksum_after, checksum_before)
        self.assertIn(
            f"  ./{subject.NOMINAL_FORWARD_NAME}\n".encode("utf-8"),
            checksum_after)
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            subject.finalize_nominal_forward(
                self.dataset, output,
                version="human-v2", mounting_id="rig-2026",
                review_manifest=self._review_manifest(("f0002",)),
                bearing_camera_cw_deg=None, panorama_column=180.0,
                uncertainty_deg=1.0, operator="Human Reviewer",
                approved_at="2026-08-24T16:00:00Z", notes="second",
                evidence_frame_ids=("f0002",),
                approve_as_authority=True)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(
            (self.dataset / checksums.CHECKSUM_FILE).read_bytes(),
            checksum_after)

    def test_finalize_from_column_and_refuses_implicit_or_unknown_evidence(self):
        with self.assertRaisesRegex(subject.ReviewError, "explicit"):
            subject.finalize_nominal_forward(
                self.dataset, self.root / "implicit.json",
                version="v1", mounting_id="rig",
                review_manifest=self.root / "unused.json",
                bearing_camera_cw_deg=None, panorama_column=90.0,
                uncertainty_deg=2.0, operator="reviewer",
                approved_at="2026-08-24T12:00:00Z", notes="reviewed",
                evidence_frame_ids=("f0000",),
                approve_as_authority=False)
        with self.assertRaisesRegex(subject.ReviewError, "unknown evidence"):
            subject.finalize_nominal_forward(
                self.dataset, self.root / "unknown.json",
                version="v1", mounting_id="rig",
                review_manifest=self.root / "unused.json",
                bearing_camera_cw_deg=None, panorama_column=90.0,
                uncertainty_deg=2.0, operator="reviewer",
                approved_at="2026-08-24T12:00:00Z", notes="reviewed",
                evidence_frame_ids=("f9999",),
                approve_as_authority=True)
        output = self.dataset / subject.NOMINAL_FORWARD_NAME
        document = subject.finalize_nominal_forward(
            self.dataset, output,
            version="v1", mounting_id="rig",
            review_manifest=self._review_manifest(("f0000",)),
            bearing_camera_cw_deg=None, panorama_column=90.0,
            uncertainty_deg=2.0, operator="reviewer",
            approved_at="2026-08-24T12:00:00Z", notes="reviewed",
            evidence_frame_ids=("f0000",),
            approve_as_authority=True)
        self.assertEqual(document["panorama_column"], 90.0)
        self.assertEqual(document["bearing_camera_cw_deg"], 270.0)
        checksums.verify(self.dataset)

    def test_finalize_requires_dataset_root_and_refuses_stale_manifest(self):
        arguments = dict(
            version="v1", mounting_id="rig",
            review_manifest=self._review_manifest(("f0000",)),
            bearing_camera_cw_deg=0.0, panorama_column=None,
            uncertainty_deg=1.0, operator="reviewer",
            approved_at="2026-08-24T12:00:00Z", notes="reviewed",
            evidence_frame_ids=("f0000",), approve_as_authority=True)
        with self.assertRaisesRegex(subject.ReviewError, "checksummed dataset"):
            subject.finalize_nominal_forward(
                self.dataset, self.root / "external.json", **arguments)
        checksum = self.dataset / checksums.CHECKSUM_FILE
        checksum.write_text("stale\n")
        with self.assertRaisesRegex(subject.ReviewError, "stale pre-finalization"):
            subject.finalize_nominal_forward(
                self.dataset,
                self.dataset / subject.NOMINAL_FORWARD_NAME,
                **arguments)
        self.assertFalse(
            (self.dataset / subject.NOMINAL_FORWARD_NAME).exists())
        self.assertEqual(checksum.read_text(), "stale\n")

    def test_checksum_failure_rolls_back_record_and_manifest(self):
        output = self.dataset / subject.NOMINAL_FORWARD_NAME
        checksum = self.dataset / checksums.CHECKSUM_FILE
        original_checksum = checksum.read_bytes()
        with mock.patch.object(
                subject.checksums, "regenerate",
                side_effect=OSError("simulated checksum publication failure")):
            with self.assertRaisesRegex(OSError, "simulated checksum"):
                subject.finalize_nominal_forward(
                    self.dataset, output,
                    version="v1", mounting_id="rig",
                    review_manifest=self._review_manifest(("f0000",)),
                    bearing_camera_cw_deg=0.0, panorama_column=None,
                    uncertainty_deg=1.0, operator="reviewer",
                    approved_at="2026-08-24T12:00:00Z", notes="reviewed",
                    evidence_frame_ids=("f0000",),
                    approve_as_authority=True)
        self.assertFalse(output.exists())
        self.assertEqual(checksum.read_bytes(), original_checksum)
        self.assertFalse(subject._transaction_path(self.dataset).exists())
        checksums.verify(self.dataset)

    def test_interrupted_record_checksum_gap_rolls_forward(self):
        class SimulatedProcessDeath(BaseException):
            pass

        output = self.dataset / subject.NOMINAL_FORWARD_NAME
        review_manifest = self._review_manifest(("f0000",))
        old_checksum = (self.dataset / checksums.CHECKSUM_FILE).read_bytes()
        with mock.patch.object(
                subject.checksums, "regenerate",
                side_effect=SimulatedProcessDeath()):
            with self.assertRaises(SimulatedProcessDeath):
                subject.finalize_nominal_forward(
                    self.dataset, output,
                    version="v1", mounting_id="rig",
                    review_manifest=review_manifest,
                    bearing_camera_cw_deg=0.0, panorama_column=None,
                    uncertainty_deg=1.0, operator="reviewer",
                    approved_at="2026-08-24T12:00:00Z", notes="reviewed",
                    evidence_frame_ids=("f0000",),
                    approve_as_authority=True)
        self.assertTrue(output.is_file())
        self.assertEqual(
            (self.dataset / checksums.CHECKSUM_FILE).read_bytes(),
            old_checksum)
        self.assertTrue(subject._transaction_path(self.dataset).is_dir())
        subject._recover_finalize_transaction(self.dataset)
        self.assertFalse(subject._transaction_path(self.dataset).exists())
        checksums.verify(self.dataset)
        nominal_forward.load(output, expected_dataset=self.dataset.name)

    def test_finalize_requires_exact_untampered_review_evidence(self):
        review_manifest = self._review_manifest(("f0000", "f0002"))
        arguments = dict(
            version="v1", mounting_id="rig",
            review_manifest=review_manifest,
            bearing_camera_cw_deg=0.0, panorama_column=None,
            uncertainty_deg=1.0, operator="reviewer",
            approved_at="2026-08-24T12:00:00Z", notes="reviewed",
            approve_as_authority=True)
        with self.assertRaisesRegex(subject.ReviewError, "exactly match"):
            subject.finalize_nominal_forward(
                self.dataset,
                self.dataset / subject.NOMINAL_FORWARD_NAME,
                evidence_frame_ids=("f0000",), **arguments)
        with (review_manifest.parent / "contact_sheet.png").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(subject.ReviewError, "output bytes"):
            subject.finalize_nominal_forward(
                self.dataset,
                self.dataset / subject.NOMINAL_FORWARD_NAME,
                evidence_frame_ids=("f0000", "f0002"), **arguments)
        self.assertFalse(
            (self.dataset / subject.NOMINAL_FORWARD_NAME).exists())


if __name__ == "__main__":
    unittest.main()
