"""Network-free contract tests for transactional Mapillary downloads."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from experimental.overhead_matching.swag.farfield.collection import (
    extract_stitch as subject,
)


def jpeg_bytes(width=128, height=64) -> bytes:
    image = Image.new("RGB", (width, height))
    image.putdata([
        ((x * 17 + y * 3) % 256,
         (x * 5 + y * 19) % 256,
         (x * 11 + y * 7) % 256)
        for y in range(height)
        for x in range(width)
    ])
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=95)
    return stream.getvalue()


def image_record(identifier: str, index: int) -> dict:
    return {
        "id": identifier,
        "lat": 42.0 + index * 0.001,
        "lng": -71.0 + index * 0.001,
        "compass_angle": 90.0,
        "computed_compass_angle": 0.0,
        "captured_at": 1_000 * (index + 1),
        "camera_type": "spherical",
        "width": 128,
        "height": 64,
        "sequence_id": "component",
        "downloaded": False,
    }


class FakeClient:
    def __init__(self, payload: bytes, failing=()):
        self.payload = payload
        self.failing = set(failing)
        self.requested = []

    def get_image_url(self, image_id, max_width=None):
        self.requested.append((image_id, max_width))
        return f"https://example.invalid/{image_id}/{max_width or 'original'}"

    def download_image(self, url):
        image_id = url.split("/")[-2]
        if image_id in self.failing:
            raise RuntimeError("simulated transport failure")
        return self.payload


class DownloadPublicationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "request.json"
        self.output = self.root / "raw" / "example"
        self.images = [image_record("100", 0), image_record("200", 1)]
        length_km = round(subject.haversine_m(
            self.images[0]["lat"], self.images[0]["lng"],
            self.images[1]["lat"], self.images[1]["lng"]), 3) / 1000.0
        # The production handoff is the strict seed_to_trajectory artifact,
        # rather than a permissive ad-hoc list of images.
        self.source.write_text(json.dumps({
            "metadata": {
                "area_name": "example",
                "total_sequences": 1,
                "total_images": 2,
                "total_length_km": round(length_km, 2),
                "created_at": 1,
            },
            "sequences": [{
                "id": "example",
                "length_km": round(length_km, 3),
                "image_count": 2,
                "start_time": 1_000,
                "end_time": 2_000,
                "camera_types": ["spherical"],
                "min_width": 128,
                "min_height": 64,
                "images": self.images,
            }],
            "trajectory": {
                "name": "example",
                "seed_pkey": "100",
                "seed_sequence_id": "component",
                "creator_username": "",
                "camera_type": "spherical",
                "is_equirectangular": True,
                "camera_parameters": None,
                "seed_image_count": 2,
                "seed_length_km": round(length_km, 3),
                "component_sequence_ids": ["component"],
                "chain_image_count": 2,
                "chain_length_km": round(length_km, 3),
                "window_hours": 36.0,
                "stitch_time_s": 300.0,
                "stitch_dist_m": 100.0,
            },
            "provenance": {
                "schema": "farfield_provenance/v1",
                "generator": subject.seed_to_trajectory._GENERATOR,
                "git_commit": "abc123",
                "argv": [],
                "created": "2026-01-01T00:00:00+00:00",
                "inputs": {"seed_pkey": "100"},
                "config": {
                    "name": "example",
                    "window_hours": 36.0,
                    "stitch_time_s": 300.0,
                    "stitch_dist_m": 100.0,
                },
                "notes": "",
            },
        }))
        self.jpeg = jpeg_bytes()

    def download(self, client, **overrides):
        arguments = {
            "manifest_path": self.source,
            "sequence": "example",
            "out_dir": self.output,
            "workers": 1,
            "max_width": 4096,
            "min_spacing_m": 0.0,
        }
        arguments.update(overrides)
        with mock.patch.object(
                subject, "MapillaryClient", return_value=client), \
             mock.patch.object(
                 subject.provenance, "git_commit", return_value="abc123"):
            return subject.download_sequence(**arguments)

    def test_success_publishes_exact_complete_directory_and_reuses_it(self):
        client = FakeClient(self.jpeg)
        self.assertTrue(self.download(client))
        self.assertTrue(self.output.is_dir())
        self.assertFalse(
            self.output.with_name("example.incomplete").exists())
        manifest = json.loads(
            (self.output / subject.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["schema"], subject.DOWNLOAD_SCHEMA)
        self.assertTrue(manifest["complete"])
        self.assertEqual([item["id"] for item in manifest["expected"]],
                         ["100", "200"])

        with mock.patch.object(
                subject, "MapillaryClient",
                side_effect=AssertionError("completed reuse contacted API")):
            self.assertTrue(subject.download_sequence(
                self.source, "example", self.output, 1, 4096, 0.0))

    def test_failed_download_retains_incomplete_and_resume_skips_valid_pair(self):
        first = FakeClient(self.jpeg, failing={"200"})
        self.assertFalse(self.download(first))
        incomplete = self.output.with_name("example.incomplete")
        self.assertTrue(incomplete.is_dir())
        self.assertFalse(self.output.exists())

        second = FakeClient(self.jpeg)
        self.assertTrue(self.download(second))
        self.assertEqual(second.requested, [("200", 4096)])
        self.assertFalse(incomplete.exists())
        self.assertTrue(self.output.is_dir())

    def test_corrupt_existing_pair_is_not_skipped_or_overwritten(self):
        first = FakeClient(self.jpeg, failing={"200"})
        self.assertFalse(self.download(first))
        incomplete = self.output.with_name("example.incomplete")
        first_jpg = next(incomplete.glob("100_*.jpg"))
        first_jpg.write_bytes(b"corrupt")

        second = FakeClient(self.jpeg)
        self.assertFalse(self.download(second))
        self.assertEqual(first_jpg.read_bytes(), b"corrupt")
        self.assertIn(("200", 4096), second.requested)
        self.assertNotIn(("100", 4096), second.requested)

    def test_stale_extra_file_fails_before_network(self):
        incomplete = self.output.with_name("example.incomplete")
        incomplete.mkdir(parents=True)
        (incomplete / "stale.jpg").write_bytes(self.jpeg)
        with mock.patch.object(
                subject, "MapillaryClient",
                side_effect=AssertionError("stale input contacted API")):
            with self.assertRaisesRegex(ValueError, "stale or unexpected"):
                subject.download_sequence(
                    self.source, "example", self.output, 1, 4096, 0.0)

    def test_interrupted_single_file_is_replaced_on_resume(self):
        incomplete = self.output.with_name("example.incomplete")
        incomplete.mkdir(parents=True)
        images, _ = subject._load_request(self.source, "example", 0.0)
        stem = images[0]["_stem"]
        (incomplete / f"{stem}.jpg").write_bytes(b"interrupted")

        client = FakeClient(self.jpeg)
        self.assertTrue(self.download(client))
        self.assertEqual(client.requested, [
            ("100", 4096),
            ("200", 4096),
        ])

    def test_completed_output_rejects_changed_configuration(self):
        self.assertTrue(self.download(FakeClient(self.jpeg)))
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.download(FakeClient(self.jpeg), max_width=2048)

    def test_completed_output_reuses_across_worker_count(self):
        self.assertTrue(self.download(FakeClient(self.jpeg)))
        with mock.patch.object(
                subject, "MapillaryClient",
                side_effect=AssertionError("worker-only change contacted API")):
            self.assertTrue(subject.download_sequence(
                self.source, "example", self.output, 8, 4096, 0.0))

    def test_resume_rejects_sidecar_field_changed_from_source(self):
        first = FakeClient(self.jpeg, failing={"200"})
        self.assertFalse(self.download(first))
        sidecar = next(
            self.output.with_name("example.incomplete").glob("100_*.json"))
        record = json.loads(sidecar.read_text())
        record["computed_compass_angle"] = 12.0
        sidecar.write_text(json.dumps(record))

        second = FakeClient(self.jpeg)
        self.assertFalse(self.download(second))
        self.assertNotIn(("100", 4096), second.requested)

    def test_strict_stage_one_manifest_is_required(self):
        document = json.loads(self.source.read_text())
        del document["sequences"][0]["images"][0]["downloaded"]
        self.source.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "missing"):
            self.download(FakeClient(self.jpeg))


if __name__ == "__main__":
    unittest.main()
