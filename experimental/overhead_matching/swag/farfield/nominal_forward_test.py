import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import nominal_forward


def record(**changes):
    value = {
        "schema": nominal_forward.SCHEMA,
        "frame": nominal_forward.FRAME,
        "dataset": "boston_harbor_leg1",
        "version": "v1",
        "mounting_id": "boston_leg1_mount",
        "panorama_column": 1500.0,
        "panorama_width": 2000,
        "bearing_camera_cw_deg": 90.0,
        "uncertainty_deg": 2.0,
        "evidence_frame_ids": ["f0003", "f0100"],
        "operator": "reviewer@example.com",
        "approved_at": "2026-08-23T20:00:00Z",
        "approved": True,
        "notes": "human annotation",
    }
    value.update(changes)
    return value


class NominalForwardTest(unittest.TestCase):
    def test_parse_and_rotate(self):
        calibration = nominal_forward.parse(
            record(), expected_dataset="boston_harbor_leg1")
        self.assertEqual(
            nominal_forward.camera_to_forward_cw_deg(120.0, calibration),
            30.0)

    def test_diagnostic_is_not_authority(self):
        with self.assertRaisesRegex(ValueError, "approved"):
            nominal_forward.parse(record(approved=False))

    def test_dataset_binding_is_required(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            nominal_forward.parse(record(), expected_dataset="pohang")

    def test_column_and_derived_bearing_must_agree(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            nominal_forward.parse(record(bearing_camera_cw_deg=89.0))

    def test_boolean_is_not_numeric(self):
        with self.assertRaisesRegex(ValueError, "real number"):
            nominal_forward.parse(record(uncertainty_deg=True))

    def test_fields_are_exact(self):
        missing = record()
        del missing["notes"]
        with self.assertRaisesRegex(ValueError, "missing notes"):
            nominal_forward.parse(missing)
        with self.assertRaisesRegex(ValueError, "unknown 'candidate'"):
            nominal_forward.parse(record(candidate=90.0))

    def test_notes_must_be_a_string(self):
        with self.assertRaisesRegex(ValueError, "notes must be a string"):
            nominal_forward.parse(record(notes=["not", "prose"]))

    def test_evidence_frame_ids_are_unique(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            nominal_forward.parse(
                record(evidence_frame_ids=["f0003", "f0003"]))

    def test_version_and_mounting_id_are_identifiers(self):
        calibration = nominal_forward.parse(record(version="2026-08-23"))
        self.assertEqual(calibration.version, "2026-08-23")
        for field, value in (("version", "v1/../../candidate"),
                             ("mounting_id", "rig one")):
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "must match"):
                nominal_forward.parse(record(**{field: value}))

    def test_approved_at_requires_an_aware_iso_timestamp(self):
        calibration = nominal_forward.parse(
            record(approved_at="2026-08-23T16:00:00-04:00"))
        self.assertEqual(
            calibration.approved_at, "2026-08-23T16:00:00-04:00")
        for value in ("today", "2026-08-23T20:00:00"):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "approved_at"):
                nominal_forward.parse(record(approved_at=value))

    def test_bearing_must_be_canonical(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 360\)"):
            nominal_forward.parse(record(bearing_camera_cw_deg=450.0))

    def test_load_rejects_non_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nominal_forward.json"
            path.write_text(json.dumps([]))
            with self.assertRaisesRegex(ValueError, "must be an object"):
                nominal_forward.load(path)

    def test_load_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nominal_forward.json"
            path.write_text('{"schema":"first","schema":"second"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                nominal_forward.load(path)

    def test_load_rejects_nonfinite_json_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nominal_forward.json"
            path.write_text(json.dumps(record()).replace(
                '"uncertainty_deg": 2.0', '"uncertainty_deg": NaN'))
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                nominal_forward.load(path)


if __name__ == "__main__":
    unittest.main()
