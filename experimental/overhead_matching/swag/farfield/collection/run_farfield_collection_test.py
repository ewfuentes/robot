"""Network-free contract tests for the collection orchestrator."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield.collection import (
    run_farfield_collection as runner,
)


class ResolveStageBoundaryTest(unittest.TestCase):

    @staticmethod
    def args(root: Path):
        return SimpleNamespace(
            manifest_dir=root,
            window_hours=36.0,
            stitch_time=300.0,
            stitch_dist=100.0,
            workers=4,
            dry_run=False,
        )

    @staticmethod
    def document():
        return {
            "provenance": {
                "config": {
                    "name": "example",
                    "window_hours": 36.0,
                    "stitch_time_s": 300.0,
                    "stitch_dist_m": 100.0,
                },
            },
        }

    def test_existing_manifest_is_strictly_validated_before_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "example.json"
            manifest.write_text("{}")
            args = self.args(root)
            with mock.patch.object(
                    runner.seed_to_trajectory, "validate_sequence_manifest",
                    return_value=self.document()) as validate, mock.patch.object(
                        runner, "run_module") as invoke:
                self.assertTrue(runner.stage_resolve(
                    "example", {"seed_pkey": "seed-1"}, args))

        validate.assert_called_once_with(
            manifest,
            expected_sequence_id="example",
            expected_seed_pkey="seed-1",
        )
        invoke.assert_not_called()

    def test_existing_manifest_with_different_recipe_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example.json").write_text("{}")
            args = self.args(root)
            document = self.document()
            document["provenance"]["config"]["stitch_time_s"] = 301.0
            with mock.patch.object(
                    runner.seed_to_trajectory, "validate_sequence_manifest",
                    return_value=document), mock.patch.object(
                        runner, "run_module") as invoke:
                self.assertFalse(runner.stage_resolve(
                    "example", {"seed_pkey": "seed-1"}, args))

        invoke.assert_not_called()

    def test_invalid_completed_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example.json").write_text("{}")
            args = self.args(root)
            with mock.patch.object(
                    runner.seed_to_trajectory, "validate_sequence_manifest",
                    side_effect=ValueError("bad manifest")), mock.patch.object(
                        runner, "run_module") as invoke:
                self.assertFalse(runner.stage_resolve(
                    "example", {"seed_pkey": "seed-1"}, args))

        invoke.assert_not_called()

    def test_incomplete_manifest_blocks_network_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example.json.incomplete").write_text("partial")
            args = self.args(root)
            with mock.patch.object(runner, "run_module") as invoke:
                self.assertFalse(runner.stage_resolve(
                    "example", {"seed_pkey": "seed-1"}, args))

        invoke.assert_not_called()


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

    def test_finish_records_direct_full_pbf_geometry_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                catalog_sources_base=root / "sources",
                catalog_version="full_v1",
                dedupe_tolerance_m=10.0,
            )
            with mock.patch.object(runner, "_write_provenance"), \
                 mock.patch.object(
                     runner, "_publish_full_catalog",
                     return_value=True) as publish:
                self.assertTrue(runner._finish_landmark_stage(
                    "example", root / "selected.feather", [root / "x.pbf"],
                    (-71.2, 41.8, -70.8, 42.2),
                    ["north-america/us/massachusetts-latest.osm.pbf"],
                    None, False, args,
                    {"schema": "farfield_catalog_source_coverage/v2",
                     "status": "passed", "message": "complete",
                     "details": []}))

        config = publish.call_args.kwargs["config"]
        self.assertEqual(
            config["osm_geometry_index_mode"],
            "full_pbf_complete_geometry_index")
        self.assertNotIn("osm_preextract_strategy", config)

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
                with mock.patch.object(
                        runner.shutil, "copyfile",
                        side_effect=AssertionError("must reuse")):
                    self.assertTrue(runner._publish_full_catalog(
                        "example", source, args,
                        {"bbox_wsen": [1, 2, 3, 4]}))

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


class PinholePublicationTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "datasets" / "example"
        panorama = self.dataset / "panorama"
        panorama.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (16, 8), (index * 20, 40, 60)).save(
                panorama / f"f{index:04d}.jpg")
        (self.dataset / "pipeline_metadata.json").write_text(
            '{"is_equirectangular": true}\n')
        (self.dataset / "frames_gps.csv").write_text("frame_file\n")
        self.args = SimpleNamespace(
            output_base=self.root / "datasets",
            pinhole_base=(self.root / "artifacts" /
                          paths_lib.PINHOLE_IMAGES),
            pinhole_version="v1",
            pinhole_res=8,
            convert_workers=3,
            dry_run=False,
        )

    def render_outputs(self, target, argv, description, args):
        del target, description, args
        output = Path(argv[1])
        for stem in ("f0000", "f0001"):
            directory = output / stem
            directory.mkdir(parents=True)
            for face in runner.PINHOLE_FACES:
                Image.new("RGB", (8, 8), (10, 20, 30)).save(
                    directory / f"{face}.jpg")
        return True

    def test_publishes_complete_typed_directory_and_refuses_clobber(self):
        with mock.patch.object(
                runner, "run_bazel", side_effect=self.render_outputs) as invoke, \
             mock.patch.object(runner.publication.indexes, "refresh"):
            self.assertTrue(runner.stage_pinhole(
                "example", {"pano": True}, self.args))

        output = self.args.pinhole_base / "example" / "v1"
        artifact.open_artifact(
            output, expected_kind=paths_lib.PINHOLE_IMAGES,
            expected_dataset="example", expected_version="v1")
        manifest = artifact.load_manifest(output)
        panorama_keys = ["f0000", "f0001"]
        self.assertEqual(
            manifest.declared_outputs,
            paths_lib.pinhole_declared_outputs(panorama_keys))
        self.assertEqual(
            dict(manifest.config),
            paths_lib.pinhole_manifest_config(
                paths_lib.dataset_source_digests(self.dataset),
                resolution=8, panorama_keys=panorama_keys))
        self.assertNotIn("orchestration", manifest.config)
        self.assertNotIn("build_identity", manifest.config)
        self.assertFalse(output.with_name("v1.incomplete").exists())
        self.assertEqual(invoke.call_args.args[1][1],
                         output.with_name("v1.incomplete"))

        with mock.patch.object(runner, "run_bazel") as second_invoke:
            self.assertFalse(runner.stage_pinhole(
                "example", {"pano": True}, self.args))
        second_invoke.assert_not_called()

    def test_partial_render_is_never_published(self):
        def partial(target, argv, description, args):
            del target, description, args
            output = Path(argv[1]) / "f0000"
            output.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(output / "yaw_000.jpg")
            return True

        with mock.patch.object(runner, "run_bazel", side_effect=partial):
            self.assertFalse(runner.stage_pinhole(
                "example", {"pano": True}, self.args))

        output = self.args.pinhole_base / "example" / "v1"
        self.assertFalse(output.exists())
        self.assertTrue(output.with_name("v1.incomplete").is_dir())

    def test_dry_run_does_not_create_publication_directories(self):
        self.args.dry_run = True
        with mock.patch.object(runner, "run_bazel", return_value=True) as invoke:
            self.assertTrue(runner.stage_pinhole(
                "example", {"pano": True}, self.args))
        self.assertEqual(
            invoke.call_args.args[1][1],
            self.args.pinhole_base / "example" / "v1.incomplete")
        self.assertFalse(self.args.pinhole_base.exists())


class LandmarkOrchestrationTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(
            output_base=self.root / "datasets",
            landmark_buffer_km=25.0,
            osm_cache_dir=self.root / "osm",
            catalog_sources_base=self.root / "catalog_sources",
            catalog_version="full_v1",
            extract_mem_cap_gb=12,
            enc_root=self.root / "enc",
            dedupe_tolerance_m=10.0,
            dry_run=False,
        )
        dataset_dir = self.args.output_base / "example"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "pano_id_mapping.csv").write_text("id\n")
        self.args.osm_cache_dir.mkdir()

    def pbf(self, stem):
        path = self.args.osm_cache_dir / f"{stem}-260824.osm.pbf"
        path.write_bytes(b"pbf")
        return path

    def test_failed_coverage_stops_before_extraction(self):
        self.pbf("massachusetts")
        config = {
            "osm": "north-america/us/massachusetts-latest.osm.pbf",
            "enc_state": None,
        }
        with mock.patch.object(
                runner, "bbox_from_dataset",
                return_value=(-71.2, 41.8, -70.8, 42.2)), mock.patch.object(
                    runner, "check_coverage",
                    return_value=(False, "1.0% of mappable land is uncovered", [])), \
             mock.patch.object(runner, "run_bazel") as invoke:
            self.assertFalse(
                runner.stage_landmarks("example", config, self.args))

        invoke.assert_not_called()

    def test_dry_run_performs_no_landmark_io(self):
        self.args.dry_run = True
        config = {
            "osm": "north-america/us/massachusetts-latest.osm.pbf",
            "enc_state": "MA",
        }
        with mock.patch.object(
                runner, "bbox_from_dataset") as bbox, mock.patch.object(
                    runner, "check_coverage") as coverage, mock.patch.object(
                        runner, "run_bazel") as invoke:
            self.assertTrue(
                runner.stage_landmarks("example", config, self.args))
        bbox.assert_not_called()
        coverage.assert_not_called()
        invoke.assert_not_called()

    def run_stage(self, config, selection=None):
        with mock.patch.object(
                runner, "bbox_from_dataset",
                return_value=(-71.2, 41.8, -70.8, 42.2)), \
             mock.patch.object(
                 runner, "check_coverage",
                 return_value=(True, "complete", [])), \
             mock.patch.object(
                 runner, "run_bazel", return_value=True) as invoke, \
             mock.patch.object(
                 runner, "_finish_landmark_stage",
                 return_value=True) as finish, \
             mock.patch.object(
                 runner.download_enc_cells, "validate_selection",
                 return_value=selection) as validate:
            result = runner.stage_landmarks("example", config, self.args)
        return result, invoke, finish, validate

    def test_single_pbf_no_enc_flow(self):
        pbf = self.pbf("massachusetts")
        result, invoke, finish, validate = self.run_stage({
            "osm": "north-america/us/massachusetts-latest.osm.pbf",
            "enc_state": None,
        })
        self.assertTrue(result)
        self.assertEqual(invoke.call_count, 1)
        target, argv = invoke.call_args.args[:2]
        self.assertEqual(
            target,
            runner.DATASET_TOOLS + ":extract_landmarks_from_osm")
        self.assertEqual(argv[:2], ["--pbf_file", pbf])
        self.assertNotIn("--node_margin_deg", argv)
        validate.assert_not_called()
        finish.assert_called_once()

    def test_multi_pbf_flow_merges_all_exact_outputs(self):
        first = self.pbf("massachusetts")
        second = self.pbf("rhode-island")
        result, invoke, finish, _ = self.run_stage({
            "osm": [
                "north-america/us/massachusetts-latest.osm.pbf",
                "north-america/us/rhode-island-latest.osm.pbf",
            ],
            "enc_state": None,
        })
        self.assertTrue(result)
        self.assertEqual(invoke.call_count, 3)
        self.assertEqual(invoke.call_args_list[0].args[1][:2],
                         ["--pbf_file", first])
        self.assertEqual(invoke.call_args_list[1].args[1][:2],
                         ["--pbf_file", second])
        merge_target, merge_argv = invoke.call_args_list[2].args[:2]
        self.assertEqual(
            merge_target,
            runner.DATASET_TOOLS + ":merge_landmark_feathers")
        self.assertEqual(merge_argv[0], "--inputs")
        self.assertEqual(len(merge_argv[1:3]), 2)
        finish.assert_called_once()

    def test_enc_flow_uses_only_exact_invocation_selection(self):
        self.pbf("massachusetts")
        selection = {
            "catalog_state": "MA",
            "bbox": [-71.2, 41.8, -70.8, 42.2],
            "band": 5,
            "explicit_cells": False,
            "cells": ["US5BOSCD", "US5BOSDD"],
        }
        result, invoke, finish, validate = self.run_stage({
            "osm": "north-america/us/massachusetts-latest.osm.pbf",
            "enc_state": "MA",
        }, selection=selection)
        self.assertTrue(result)
        self.assertEqual(invoke.call_count, 4)
        download_target, download_argv = invoke.call_args_list[1].args[:2]
        self.assertEqual(
            download_target,
            runner.DATASET_TOOLS + ":download_enc_cells")
        self.assertIn("--band", download_argv)
        self.assertIn("--selection_output", download_argv)
        selection_path = (
            self.args.catalog_sources_base / "example" / "full_v1" /
            "enc_example_selection.json"
        )
        validate.assert_called_once_with(selection_path, self.args.enc_root)

        extraction_target, extraction_argv = invoke.call_args_list[2].args[:2]
        self.assertEqual(
            extraction_target,
            runner.DATASET_TOOLS + ":extract_landmarks_from_enc")
        selection_index = extraction_argv.index("--selection")
        self.assertEqual(
            extraction_argv[selection_index + 1], selection_path)
        self.assertNotIn("--cells", extraction_argv)
        finish.assert_called_once()


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
