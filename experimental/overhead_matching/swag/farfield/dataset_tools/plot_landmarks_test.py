"""Hermetic contract tests for typed catalog coverage diagnostics."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shapely.geometry import Point

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    plot_landmarks as subject,
)


DATASET = "example"
SPEC = "north-america/us/massachusetts-latest.osm.pbf"
ANALYSIS_CONFIG = {
    "grid_cells": 4,
    "max_empty_run": 4,
    "empty_fraction_warning": 1.0,
    "far_range_km": 1.0,
    "min_far_fraction": 0.0,
    "max_track_samples": 10,
}


def write_dataset(root: Path, dataset_name: str = DATASET) -> Path:
    dataset_dir = root / "datasets" / dataset_name
    panorama = dataset_dir / "panorama"
    panorama.mkdir(parents=True)
    (dataset_dir / "pipeline_metadata.json").write_text(json.dumps({
        "dataset_name": dataset_name,
    }))
    (dataset_dir / "frames_gps.csv").write_text(
        "idx,latitude,longitude,dist_m,video_t_s\n"
        "0,42.00,-71.00,0,0\n"
        "1,42.10,-70.90,1000,10\n"
        "2,42.20,-70.80,2000,20\n"
    )
    for name in (
            "f0000,42.000000,-71.000000,.jpg",
            "f0001,42.100000,-70.900000,.jpg",
            "f0002,42.200000,-70.800000,.jpg"):
        (panorama / name).write_bytes(b"fixture")
    return dataset_dir


def write_poly(root: Path) -> Path:
    cache = root / "raw_material" / "osm" / "poly"
    cache.mkdir(parents=True)
    subject.pbf_coverage.poly_cache_path(
        "north-america/us/massachusetts-latest.osm.pbf", cache).write_text(
        "massachusetts\n"
        "1\n"
        "  -71.3 41.8\n"
        "  -70.6 41.8\n"
        "  -70.6 42.5\n"
        "  -71.3 42.5\n"
        "  -71.3 41.8\n"
        "END\n"
        "END\n"
    )
    return cache


def catalog_frame(*, outside: bool = False):
    if outside:
        points = [Point(-72.0, 43.0)]
    else:
        points = [
            Point(-71.1, 42.0),
            Point(-70.8, 42.0),
            Point(-71.1, 42.3),
            Point(-70.8, 42.3),
        ]
    return schema.build_frame(
        ids=[f"osm:node:{index + 1}" for index in range(len(points))],
        geometries=points,
        landmark_types=["osm"] * len(points),
        tags=[
            {"man_made": "lighthouse", "name": f"Light {index + 1}"}
            for index in range(len(points))
        ],
    )


def publish_full_catalog(root: Path, *, dataset_name: str = DATASET,
                         outside: bool = False,
                         coverage_status: str = "passed") -> tuple[
                             Path, artifact.ArtifactRef]:
    frame = catalog_frame(outside=outside)
    loose = root / "raw_material" / f"{dataset_name}.feather"
    loose.parent.mkdir(parents=True, exist_ok=True)
    frame.to_feather(loose)
    output = root / "artifacts" / paths_lib.CATALOGS / dataset_name / "full_v1"
    config = {
        "schema": schema.FULL_ARTIFACT_SCHEMA,
        "bbox_wsen": [-71.2, 41.9, -70.7, 42.4],
        "osm_specs": [SPEC],
        "enc_state": None,
        "enc_cells": [],
        "enc_available": False,
        "enc_selection": None,
        "dedupe_tolerance_m": 10.0,
        "osm_preextract_strategy": "smart",
        "selected_source_feather": str(loose.resolve()),
        "selected_source_sha256": artifact.sha256_file(loose),
        "rows": len(frame),
        "source_coverage": {
            "schema": "farfield_catalog_source_coverage/v2",
            "status": coverage_status,
            "message": "fixture source coverage passed",
            "details": [],
        },
    }
    with artifact.ArtifactDirectoryBuilder(
            output,
            kind=paths_lib.CATALOGS,
            dataset=dataset_name,
            version="full_v1",
            generator="test",
            git_commit="abc123",
            config=config,
            declared_outputs=("catalog.feather",)) as builder:
        shutil.copyfile(loose, builder.output_path("catalog.feather"))
    return output, artifact.open_artifact(output)


def snapshot(directory: Path) -> dict:
    return {
        path.relative_to(directory).as_posix(): artifact.sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


class PublicationTest(unittest.TestCase):

    def setup_inputs(self, root: Path, *, outside: bool = False):
        dataset_dir = write_dataset(root)
        poly_cache = write_poly(root)
        catalog_dir, catalog_ref = publish_full_catalog(
            root, outside=outside)
        resolved = subject.load_inputs(
            DATASET, dataset_dir, catalog_dir, poly_cache)
        output = (
            root / "artifacts" / paths_lib.CATALOG_COVERAGE /
            DATASET / "coverage_v1")
        return dataset_dir, catalog_ref, resolved, output

    def test_publishes_png_report_and_review_page_without_mutating_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, catalog_ref, resolved, output = self.setup_inputs(root)
            before = snapshot(dataset_dir)
            with mock.patch.object(
                    subject.publication.indexes, "refresh"):
                reference, report = subject.publish(
                    resolved, output, ANALYSIS_CONFIG,
                    arguments=("--dataset", DATASET))
            after = snapshot(dataset_dir)
            manifest = artifact.load_manifest(output)
            persisted = json.loads(
                (output / "coverage_report.json").read_text())
            png_size = len((output / "landmark_coverage.png").read_bytes())
            html = (output / "index.html").read_text()

        self.assertTrue(report["passed"])
        self.assertEqual(before, after)
        self.assertEqual(reference.kind, paths_lib.CATALOG_COVERAGE)
        self.assertEqual(manifest.upstreams, (catalog_ref,))
        self.assertEqual(manifest.declared_outputs, subject.OUTPUTS)
        self.assertEqual(manifest.config["schema"], subject.PAYLOAD_SCHEMA)
        self.assertEqual(persisted, report)
        self.assertGreater(png_size, 100)
        self.assertIn("coverage_report.json", html)

    def test_failed_gap_check_is_published_for_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, resolved, output = self.setup_inputs(root, outside=True)
            with mock.patch.object(
                    subject.publication.indexes, "refresh"):
                _, report = subject.publish(
                    resolved, output, ANALYSIS_CONFIG)
            artifact.open_artifact(
                output, expected_kind=paths_lib.CATALOG_COVERAGE)

        self.assertFalse(report["passed"])
        self.assertIn(
            "no_landmarks",
            {finding["code"] for finding in report["findings"]},
        )

    def test_completed_diagnostic_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, resolved, output = self.setup_inputs(root)
            with mock.patch.object(
                    subject.publication.indexes, "refresh"):
                subject.publish(resolved, output, ANALYSIS_CONFIG)
                before = snapshot(output)
                with self.assertRaises(artifact.ArtifactExistsError):
                    subject.publish(resolved, output, ANALYSIS_CONFIG)
                after = snapshot(output)
        self.assertEqual(before, after)

    def test_dataset_change_after_load_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, _, resolved, output = self.setup_inputs(root)
            with (dataset_dir / "frames_gps.csv").open("a") as stream:
                stream.write("3,42.3,-70.7,3000,30\n")
            with self.assertRaisesRegex(
                    subject.CoverageError, "changed during coverage"):
                subject.publish(resolved, output, ANALYSIS_CONFIG)
            self.assertFalse(output.exists())


class InputContractTest(unittest.TestCase):

    def test_skipped_source_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = write_dataset(root)
            poly_cache = write_poly(root)
            catalog_dir, _ = publish_full_catalog(
                root, coverage_status="skipped_by_operator")
            with self.assertRaisesRegex(
                    subject.CoverageError, "must attest status='passed'"):
                subject.load_inputs(
                    DATASET, dataset_dir, catalog_dir, poly_cache)

    def test_catalog_dataset_must_match_explicit_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = write_dataset(root)
            poly_cache = write_poly(root)
            catalog_dir, _ = publish_full_catalog(
                root, dataset_name="another")
            with self.assertRaisesRegex(
                    subject.CoverageError, "invalid full catalog artifact"):
                subject.load_inputs(
                    DATASET, dataset_dir, catalog_dir, poly_cache)

    def test_missing_cached_clip_boundary_is_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = write_dataset(root)
            poly_cache = root / "empty_poly_cache"
            poly_cache.mkdir()
            catalog_dir, _ = publish_full_catalog(root)
            with self.assertRaisesRegex(
                    subject.CoverageError, "cached Geofabrik boundary"):
                subject.load_inputs(
                    DATASET, dataset_dir, catalog_dir, poly_cache)

    def test_analysis_thresholds_are_explicit_and_validated(self):
        invalid = dict(ANALYSIS_CONFIG, max_empty_run=5)
        with self.assertRaisesRegex(
                subject.CoverageError, "cannot exceed grid_cells"):
            subject.validate_analysis_config(invalid)


if __name__ == "__main__":
    unittest.main()
