import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shapely.geometry import Point

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    merge_landmark_feathers as subject,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    source_publication,
)


def write_frame(path: Path, source: str, identifier: str, x: float) -> None:
    tags = {"man_made": "tower"}
    if source == "enc":
        tags["object_class"] = "LNDMRK"
    frame = schema.build_frame(
        ids=[identifier], geometries=[Point(x, 42.0)],
        landmark_types=[source], tags=[tags],
    )
    if source == "enc":
        frame["object_class"] = ["LNDMRK"]
    frame.to_feather(path)


class MergePublicationTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.osm = self.root / "osm.feather"
        self.enc = self.root / "enc.feather"
        write_frame(self.osm, "osm", "('node', 1)", -71.0001)
        write_frame(self.enc, "enc", "('enc', 'A')", -71.0)
        self.output = self.root / "merged_v1"

    def merge(self):
        return subject.main([self.osm, self.enc], self.output, 20.0, 150.0)

    def test_publishes_strict_payload_last_with_exact_input_digests(self):
        result = self.merge()
        feather, sidecar, staging = source_publication.output_paths(self.output)
        published = schema.read_frame(feather)
        self.assertEqual(published["id"].tolist(), result["id"].tolist())
        self.assertEqual(published["object_class"].tolist(), ["LNDMRK"])
        self.assertEqual(schema.tag_dicts(published)[0]["object_class"],
                         "LNDMRK")
        record = json.loads(sidecar.read_text())
        self.assertEqual(record["schema"],
                         source_publication.SOURCE_PROVENANCE_SCHEMA)
        self.assertTrue(record["complete"])
        self.assertEqual(record["output_sha256"],
                         artifact.sha256_file(feather))
        self.assertEqual(record["input_digests"], [
            {"path": str(self.osm.resolve()),
             "sha256": artifact.sha256_file(self.osm)},
            {"path": str(self.enc.resolve()),
             "sha256": artifact.sha256_file(self.enc)},
        ])
        self.assertFalse(staging.exists())

    def test_exact_existing_output_is_reused_without_merging(self):
        expected = self.merge()
        feather, sidecar, _ = source_publication.output_paths(self.output)
        before = (feather.read_bytes(), sidecar.read_bytes())
        with mock.patch.object(
                subject, "merge_feathers",
                side_effect=AssertionError("must reuse")):
            reused = self.merge()
        self.assertEqual(reused["id"].tolist(), expected["id"].tolist())
        self.assertEqual((feather.read_bytes(), sidecar.read_bytes()), before)

    def test_changed_input_fails_closed_against_completed_output(self):
        self.merge()
        feather, sidecar, _ = source_publication.output_paths(self.output)
        before = (feather.read_bytes(), sidecar.read_bytes())
        write_frame(self.osm, "osm", "('node', 7)", -70.5)
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            self.merge()
        self.assertEqual((feather.read_bytes(), sidecar.read_bytes()), before)

    def test_input_change_during_merge_publishes_nothing(self):
        def mutate(*_args, **_kwargs):
            self.osm.write_bytes(self.osm.read_bytes() + b"changed")

        with mock.patch.object(subject, "report_cross_source_collisions",
                               side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "changed during merge"):
                self.merge()
        feather, sidecar, _ = source_publication.output_paths(self.output)
        self.assertFalse(feather.exists())
        self.assertFalse(sidecar.exists())

    def test_sidecar_only_crash_state_is_validated_and_recoverable(self):
        feather, sidecar, staging = source_publication.output_paths(self.output)
        staging.mkdir()
        staged_feather = staging / "catalog.feather"
        staged_sidecar = staging / "provenance.json"
        frame = schema.build_frame(
            ids=["('node', 9)"], geometries=[Point(-71.0, 42.0)],
            landmark_types=["osm"], tags=[{"man_made": "tower"}],
        )
        frame.to_feather(staged_feather)
        artifact.atomic_write_json(staged_sidecar, {
            "schema": source_publication.SOURCE_PROVENANCE_SCHEMA,
            "output": str(feather),
            "output_sha256": artifact.sha256_file(staged_feather),
            "rows_out": 1,
            "complete": True,
        })
        sidecar.hardlink_to(staged_sidecar)

        self.merge()
        self.assertTrue(feather.is_file())
        self.assertTrue(sidecar.is_file())
        self.assertFalse(staging.exists())


if __name__ == "__main__":
    unittest.main()
