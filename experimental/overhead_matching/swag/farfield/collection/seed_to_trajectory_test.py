import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield.collection import seed_to_trajectory
from experimental.overhead_matching.swag.farfield.collection.models import (
    PanoImage,
    PanoSequence,
)


class SequenceManifestTest(unittest.TestCase):

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    @staticmethod
    def _image(image_id, sequence_id, lng, captured_at):
        return PanoImage(
            id=image_id,
            lat=42.0,
            lng=lng,
            compass_angle=91.0,
            computed_compass_angle=90.0,
            captured_at=captured_at,
            camera_type="spherical",
            height=2048,
            width=4096,
            sequence_id=sequence_id,
            downloaded=False,
            camera_parameters=[0.5, 0.01, -0.01],
            is_pano=True,
            creator_username="collector",
            geometry_source="computed",
        )

    def _inputs(self):
        seed_component = PanoSequence(
            id="component-a",
            images=[
                self._image("seed-image", "component-a", -71.0, 1_000),
                self._image("image-a2", "component-a", -70.999, 2_000),
            ],
        )
        seed_component.compute_length()
        sequence = PanoSequence(
            id="test-trip",
            images=[
                *seed_component.images,
                self._image("image-b1", "component-b", -70.998, 3_000),
            ],
        )
        sequence.compute_length()
        trajectory = {
            "name": "test-trip",
            "seed_pkey": "seed-image",
            "seed_sequence_id": "component-a",
            "creator_username": "collector",
            "camera_type": "spherical",
            "is_equirectangular": True,
            "camera_parameters": [0.5, 0.01, -0.01],
            "seed_image_count": 2,
            "seed_length_km": round(seed_component.length_km, 3),
            "component_sequence_ids": ["component-a", "component-b"],
            "chain_image_count": 3,
            "chain_length_km": round(sequence.length_km, 3),
            "window_hours": 36.0,
            "stitch_time_s": 300.0,
            "stitch_dist_m": 100.0,
        }
        provenance = {
            "schema": "farfield_provenance/v1",
            "generator": seed_to_trajectory._GENERATOR,
            "git_commit": "deadbeef",
            "argv": ["seed_to_trajectory", "--name", "test-trip"],
            "created": "2026-08-24T12:00:00+00:00",
            "inputs": {"seed_pkey": "seed-image"},
            "config": {
                "name": "test-trip",
                "window_hours": 36.0,
                "stitch_time_s": 300.0,
                "stitch_dist_m": 100.0,
            },
            "notes": "",
        }
        return sequence, {"trajectory": trajectory, "provenance": provenance}

    def _write(self, path):
        sequence, extra = self._inputs()
        seed_to_trajectory.write_sequence_manifest(
            path, [sequence], area_name="test-trip", extra=extra)

    def test_durably_publishes_validated_completed_manifest(self):
        path = self.root / "test-trip.json"
        self._write(path)

        self.assertTrue(path.is_file())
        self.assertFalse(Path(f"{path}.incomplete").exists())
        manifest = seed_to_trajectory.validate_sequence_manifest(
            path,
            expected_sequence_id="test-trip",
            expected_seed_pkey="seed-image",
        )
        self.assertEqual(manifest["metadata"]["total_images"], 3)

    def test_refuses_completed_and_incomplete_output_without_overwriting(self):
        completed = self.root / "completed.json"
        self._write(completed)
        original = completed.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "completed manifest"):
            self._write(completed)
        self.assertEqual(completed.read_bytes(), original)

        blocked = self.root / "blocked.json"
        residue = Path(f"{blocked}.incomplete")
        residue.write_bytes(b"diagnostic residue")
        with self.assertRaisesRegex(FileExistsError, "incomplete manifest"):
            self._write(blocked)
        self.assertFalse(blocked.exists())
        self.assertEqual(residue.read_bytes(), b"diagnostic residue")

    def test_publication_failure_retains_unmistakable_valid_residue(self):
        path = self.root / "failed.json"
        with mock.patch.object(
                seed_to_trajectory.os, "link",
                side_effect=FileExistsError("concurrent winner")):
            with self.assertRaisesRegex(FileExistsError, "concurrent winner"):
                self._write(path)

        residue = Path(f"{path}.incomplete")
        self.assertFalse(path.exists())
        self.assertTrue(residue.is_file())
        with self.assertRaisesRegex(ValueError, "incomplete.*cannot be consumed"):
            seed_to_trajectory.validate_sequence_manifest(residue)
        seed_to_trajectory._validate_manifest_document(
            seed_to_trajectory._load_manifest_json(residue))

    def test_validator_rejects_duplicate_keys_and_nonfinite_numbers(self):
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"metadata": {}, "metadata": {}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key 'metadata'"):
            seed_to_trajectory.validate_sequence_manifest(duplicate)

        nonfinite = self.root / "nonfinite.json"
        self._write(nonfinite)
        payload = nonfinite.read_text(encoding="utf-8")
        payload = payload.replace('"lat": 42.0', '"lat": 1e999', 1)
        nonfinite.write_text(payload, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            seed_to_trajectory.validate_sequence_manifest(nonfinite)

    def test_validator_binds_exact_sequence_and_image_identity(self):
        path = self.root / "identity.json"
        self._write(path)
        with self.assertRaisesRegex(ValueError, "sequence identity mismatch"):
            seed_to_trajectory.validate_sequence_manifest(
                path, expected_sequence_id="different-trip")
        with self.assertRaisesRegex(ValueError, "seed image identity mismatch"):
            seed_to_trajectory.validate_sequence_manifest(
                path, expected_seed_pkey="different-image")

        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["sequences"][0]["images"][1]["id"] = "seed-image"
        duplicate_image = self.root / "duplicate-image.json"
        duplicate_image.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate image id"):
            seed_to_trajectory.validate_sequence_manifest(duplicate_image)

    def test_nonfinite_generated_manifest_fails_before_creating_residue(self):
        path = self.root / "bad-generated.json"
        sequence, extra = self._inputs()
        sequence.images[0].lat = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            seed_to_trajectory.write_sequence_manifest(
                path, [sequence], area_name="test-trip", extra=extra)
        self.assertFalse(path.exists())
        self.assertFalse(Path(f"{path}.incomplete").exists())


if __name__ == "__main__":
    unittest.main()
