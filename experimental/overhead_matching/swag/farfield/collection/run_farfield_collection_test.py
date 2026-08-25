"""Network-free contract tests for the collection orchestrator."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.collection import (
    run_farfield_collection as runner,
)


class TrimStageBoundaryTest(unittest.TestCase):

    @staticmethod
    def args(root: Path, *, matched_from=None, positive_set=None):
        return SimpleNamespace(
            catalog_base=root / "artifacts" / paths_lib.CATALOGS,
            catalog_version="full_v1",
            trimmed_catalog_version="trimmed_v1",
            trim_min_building_area_m2=2000.0,
            trim_min_building_levels=6.0,
            trim_confidence_floor=0.7,
            matched_from=matched_from,
            positive_set=positive_set,
            dry_run=False,
        )

    def test_default_sequence_includes_trim_and_catalog_coverage(self):
        self.assertEqual(
            runner.DEFAULT_STAGES, (1, 2, 3, 4, 5, 6, 7, 8, 9))

    def test_trim_needs_no_matching_or_positive_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            input_dir = (args.catalog_base / "example" /
                         args.catalog_version)
            input_dir.mkdir(parents=True)
            with mock.patch.object(runner, "run_module",
                                   return_value=True) as invoke:
                self.assertTrue(runner.stage_trim("example", None, args))

        main_fn, argv, description, passed_args = invoke.call_args.args
        self.assertIs(main_fn, runner.trim_catalog.cli)
        self.assertEqual(description, "[7 TRIM] example")
        self.assertIs(passed_args, args)
        self.assertEqual(argv, [
            "--input_catalog_dir", input_dir,
            "--output_dir", (args.catalog_base / "example" /
                             args.trimmed_catalog_version),
            "--min_building_area_m2", 2000.0,
            "--min_building_levels", 6.0,
        ])

    def test_optional_evidence_is_forwarded_only_when_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matching = root / "matches"
            positive = root / "positives.json"
            args = self.args(
                root, matched_from=[matching], positive_set=positive)
            (args.catalog_base / "example" /
             args.catalog_version).mkdir(parents=True)
            with mock.patch.object(runner, "run_module",
                                   return_value=True) as invoke:
                self.assertTrue(runner.stage_trim("example", None, args))

        argv = invoke.call_args.args[1]
        self.assertIn("--matched_from", argv)
        self.assertIn(matching, argv)
        self.assertIn("--confidence_floor", argv)
        self.assertIn("--positive_set", argv)
        self.assertIn(positive, argv)


class FullCatalogPublicationTest(unittest.TestCase):

    def test_stage5_helper_publishes_one_typed_regular_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw_material" / "catalog_sources" / "full.feather"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"compact-catalog")
            args = SimpleNamespace(
                catalog_base=root / "artifacts" / paths_lib.CATALOGS,
                catalog_version="full_v1",
            )
            frame = mock.MagicMock()
            frame.__len__.return_value = 3
            with mock.patch.object(runner.catalog_schema, "read_frame",
                                   return_value=frame), mock.patch.object(
                                       runner.publication.indexes, "refresh"):
                self.assertTrue(runner._publish_full_catalog(
                    "example", source, args, {"bbox_wsen": [1, 2, 3, 4]}))

            output = args.catalog_base / "example" / "full_v1"
            reference = artifact.open_artifact(
                output, expected_kind=paths_lib.CATALOGS,
                expected_dataset="example", expected_version="full_v1")
            manifest = artifact.load_manifest(output)
            payload = (output / "catalog.feather").read_bytes()
            payload_is_symlink = (output / "catalog.feather").is_symlink()

        self.assertEqual(payload, b"compact-catalog")
        self.assertFalse(payload_is_symlink)
        self.assertEqual(manifest.declared_outputs, ("catalog.feather",))
        self.assertEqual(
            manifest.config["schema"], runner.catalog_schema.FULL_ARTIFACT_SCHEMA)
        self.assertEqual(manifest.config["rows"], 3)
        self.assertEqual(manifest.content_digest, reference.content_digest)


class CoverageStageBoundaryTest(unittest.TestCase):

    def test_stage8_consumes_full_catalog_and_publishes_separate_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                output_base=root / "datasets",
                catalog_base=root / "artifacts" / paths_lib.CATALOGS,
                catalog_version="full_v1",
                catalog_coverage_base=(
                    root / "artifacts" / paths_lib.CATALOG_COVERAGE),
                catalog_coverage_version="coverage_v1",
                osm_cache_dir=root / "raw_material" / "osm",
                coverage_grid_cells=24,
                coverage_max_empty_run=6,
                coverage_empty_fraction_warning=0.9,
                coverage_far_range_km=5.0,
                coverage_min_far_fraction=0.02,
                coverage_max_track_samples=400,
                dry_run=False,
            )
            catalog_dir = (
                args.catalog_base / "example" / args.catalog_version)
            catalog_dir.mkdir(parents=True)
            with mock.patch.object(
                    runner, "run_module", return_value=True) as invoke:
                self.assertTrue(runner.stage_plot("example", None, args))

        main_fn, argv, description, passed_args = invoke.call_args.args
        self.assertIs(main_fn, runner.plot_landmarks.cli)
        self.assertEqual(description, "[8 PLOT] example")
        self.assertIs(passed_args, args)
        self.assertEqual(argv, [
            "--dataset", "example",
            "--dataset_dir", args.output_base / "example",
            "--catalog_dir", catalog_dir,
            "--poly_cache_dir", args.osm_cache_dir / "poly",
            "--output_dir", (
                args.catalog_coverage_base / "example" /
                args.catalog_coverage_version),
            "--grid_cells", 24,
            "--max_empty_run", 6,
            "--empty_fraction_warning", 0.9,
            "--far_range_km", 5.0,
            "--min_far_fraction", 0.02,
            "--max_track_samples", 400,
        ])


if __name__ == "__main__":
    unittest.main()
