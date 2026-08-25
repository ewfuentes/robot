import argparse
import copy
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
import shapely

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    dataset,
    geometry,
    llm_lifecycle,
    nominal_forward,
    paths as paths_lib,
    pipeline,
    stage_reuse,
    testing,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    extract_landmarks,
)
from experimental.overhead_matching.swag.farfield.calibration import (
    build_alignment_diagnostics,
)
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.localization import (
    build_export,
)
from experimental.overhead_matching.swag.farfield.matching import (
    identity_review,
    match_landmarks,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    audit_requests,
    build_bearing_observations,
)


CONFIG_PATH = Path(__file__).parent / "configs" / "harbor_example.yaml"


def example_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


class SourceConfigLoadTest(unittest.TestCase):
    def test_duplicate_nested_yaml_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text("tracking:\n  window_px: 1024\n  window_px: 2048\n")
            with self.assertRaisesRegex(
                    build_config.InvalidConfigValue, "duplicate key"):
                pipeline.load_pipeline_config(path)

    def test_non_object_yaml_is_a_clean_config_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text("- not\n- a\n- mapping\n")
            with self.assertRaisesRegex(
                    build_config.InvalidConfigValue, "top-level mapping"):
                pipeline.load_pipeline_config(path)

    def test_symlink_video_is_rejected_before_build_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "source.mp4"
            target.write_bytes(b"video")
            link = root / "video-link.mp4"
            link.symlink_to(target)
            paths = paths_lib.FarfieldPaths(
                dataset="ds", root=root, overrides={"video": link})
            with self.assertRaisesRegex(FileNotFoundError, "non-symlink"):
                pipeline._source_video_inputs(paths)

    def test_new_build_declares_an_unoccupied_post_match_review_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset_base = root / "datasets" / "ds"
            panorama = dataset_base / "panorama"
            panorama.mkdir(parents=True)
            (dataset_base / "pipeline_metadata.json").write_text("{}\n")
            (dataset_base / "frames_gps.csv").write_text(
                "keyframe_idx,lat,lon\n0,0,0\n")
            (panorama / "000000.jpg").write_bytes(b"jpeg")

            checkpoint = root / "models" / "sam.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            motion = dataset_base / "motion.csv"
            motion.write_text("keyframe_idx,east_m,north_m\n0,0,0\n")
            calibration = dataset_base / "nominal_forward.json"
            artifact.atomic_write_json(calibration, {
                "schema": pipeline.nominal_forward.SCHEMA,
                "frame": pipeline.nominal_forward.FRAME,
                "dataset": "ds",
                "version": "v1",
                "mounting_id": "test-mount",
                "panorama_column": 2.0,
                "panorama_width": 4,
                "bearing_camera_cw_deg": 0.0,
                "uncertainty_deg": 1.0,
                "evidence_frame_ids": ["0"],
                "operator": "test",
                "approved_at": "2026-08-24T12:00:00+00:00",
                "approved": True,
                "notes": "fixture",
            })
            source_config = root / "source.yaml"
            source_config.write_text("fixture: true\n")

            config = example_config()
            config["tracking"]["sam2_checkpoint"] = str(checkpoint)
            config["localization_inputs"]["motion_source"] = str(motion)
            config["localization_inputs"][
                "nominal_forward_calibration"] = str(calibration)
            review_dir = root / "reviews" / "r1"
            config["localization_inputs"]["identity_review_dir"] = str(
                review_dir)
            paths = paths_lib.FarfieldPaths(
                dataset="ds", root=root,
                versions=pipeline.versions_from_config(config),
                overrides={"dataset_base": dataset_base})
            catalog_dir = paths.catalogs
            with artifact.ArtifactDirectoryBuilder(
                    catalog_dir, kind=paths_lib.CATALOGS, dataset="ds",
                    version=config["artifacts"]["catalogs_version"],
                    generator="pipeline_test", git_commit="test", arguments=(),
                    upstreams=(), config={},
                    declared_outputs=("catalog.feather",)) as builder:
                builder.output_path("catalog.feather").write_bytes(b"catalog")

            inputs = pipeline._validate_build_inputs(
                paths, config, source_config)
            self.assertEqual(inputs["identity_review_output_dir"],
                             str(review_dir.resolve()))
            self.assertEqual(inputs["identity_review_phase"],
                             "post_match_gate")
            self.assertNotIn("identity_review_manifest_digest", inputs)

            review_dir.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "must be unoccupied"):
                pipeline._validate_build_inputs(paths, config, source_config)


class ConfigContractTest(unittest.TestCase):
    def test_example_is_exact_and_fully_resolved(self):
        config = example_config()
        pipeline.validate_pipeline_config(config)
        self.assertEqual(config["tracking"]["range"],
                         {"k_start": 0, "k_end": 235})
        self.assertEqual(
            config["localization_inputs"]["landmark_position_sigma_m"],
            25.0)
        self.assertIsNone(
            config["localization_inputs"]["identity_review_dir"])
        self.assertEqual(config["localization"]["ablation_tags"], [])
        self.assertEqual(
            config["localization"]["position_mass_radii_m"],
            [50.0, 100.0, 250.0, 500.0, 1000.0])

    def test_missing_value_is_rejected(self):
        config = example_config()
        del config["matching"]["chunk_size"]
        with self.assertRaisesRegex(build_config.MissingConfigValue,
                                    "matching.chunk_size"):
            pipeline.validate_pipeline_config(config)

    def test_unknown_or_legacy_range_shape_is_rejected(self):
        config = example_config()
        config["tracking"]["ranges"] = [["a", 0, 10], ["b", 11, 20]]
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "tracking.ranges"):
            pipeline.validate_pipeline_config(config)

    def test_reversed_single_range_is_rejected(self):
        config = example_config()
        config["tracking"]["range"] = {"k_start": 20, "k_end": 10}
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "k_start must be <="):
            pipeline.validate_pipeline_config(config)

    def test_result_shaping_bool_is_not_an_integer(self):
        config = example_config()
        config["localization"]["n_particles"] = True
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "must be int"):
            pipeline.validate_pipeline_config(config)

    def test_consumer_positive_modeling_values_reject_zero_at_creation(self):
        positive_paths = (
            "bearing_observations.bearing_sigma_deg",
            "gps_course.min_displacement_m",
            "alignment_diagnostics.sun.elevation_tolerance_deg",
            "alignment_diagnostics.sweep.coarse_step_deg",
            "alignment_diagnostics.sweep.fine_step_deg",
            "alignment_diagnostics.sweep.fine_halfwidth_deg",
            "alignment_diagnostics.sweep.max_condition",
            "localization_inputs.compatibility_clip",
            "localization_inputs.odometry_sigma_pair_m",
            "localization_inputs.displacement_gate_m",
            "localization_inputs.stationary_sigma_m",
            "localization_inputs.slow_yaw_sigma_deg",
            "localization_inputs.max_visible_range_m",
            "localization_inputs.landmark_position_sigma_m",
            "localization.association_renewal_rate",
            "localization.map_cell_size_m",
            "localization.modes.cell_size_m",
            "localization.modes.heading_cell_deg",
        )
        for path in positive_paths:
            with self.subTest(path=path):
                config = example_config()
                owner = config
                parts = path.split(".")
                for part in parts[:-1]:
                    owner = owner[part]
                owner[parts[-1]] = 0.0
                with self.assertRaisesRegex(
                        build_config.InvalidConfigValue,
                        rf"{path} must be > 0\.0"):
                    pipeline.validate_pipeline_config(config)

    def test_open_probability_boundaries_are_rejected_at_creation(self):
        cases = (
            ("localization.pi0", 0.0, r"must be > 0\.0"),
            ("localization.pi0", 1.0, r"must be < 1\.0"),
            ("localization.matcher_recall", 0.0, r"must be > 0\.0"),
            ("localization.matcher_recall", 1.0, r"must be < 1\.0"),
        )
        for path, value, message in cases:
            with self.subTest(path=path, value=value):
                config = example_config()
                owner = config
                parts = path.split(".")
                for part in parts[:-1]:
                    owner = owner[part]
                owner[parts[-1]] = value
                with self.assertRaisesRegex(
                        build_config.InvalidConfigValue, message):
                    pipeline.validate_pipeline_config(config)

    def test_consumer_nonnegative_boundaries_remain_inclusive(self):
        config = example_config()
        config["gps_course"]["smooth_window_s"] = 0.0
        config["alignment_diagnostics"]["sun"]["min_speed_mps"] = 0.0
        config["alignment_diagnostics"]["sweep"]["min_arc_deg"] = 0.0
        config["alignment_diagnostics"]["sweep"]["min_support_frac"] = 0.0
        config["localization"]["association_renewal_rate"] = 1.0
        pipeline.validate_pipeline_config(config)

    def test_diagnostic_creation_domains_match_consumer_bounds(self):
        cases = (
            ("alignment_diagnostics.sun.work_width", 1, r"must be >= 2"),
            ("alignment_diagnostics.sun.elevation_tolerance_deg", 90.1,
             r"must be <= 90\.0"),
            ("alignment_diagnostics.sweep.min_arc_deg", 360.1,
             r"must be <= 360\.0"),
        )
        for path, value, message in cases:
            with self.subTest(path=path):
                config = example_config()
                owner = config
                parts = path.split(".")
                for part in parts[:-1]:
                    owner = owner[part]
                owner[parts[-1]] = value
                with self.assertRaisesRegex(
                        build_config.InvalidConfigValue, message):
                    pipeline.validate_pipeline_config(config)

    def test_llm_transport_configuration_is_not_ambiguous(self):
        config = example_config()
        config["execution"]["llm_transport"] = "on_demand"
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "must be null"):
            pipeline.validate_pipeline_config(config)
        config["execution"]["batch_gcs_prefix"] = None
        pipeline.validate_pipeline_config(config)

    def test_proposal_shares_must_sum_to_one(self):
        config = example_config()
        config["localization"]["proposal"]["share_single"] = 0.2
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "shares must sum"):
            pipeline.validate_pipeline_config(config)

    def test_metric_radii_and_ablation_tags_are_canonical(self):
        config = example_config()
        config["localization"]["position_mass_radii_m"] = [100.0, 50.0]
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "sorted"):
            pipeline.validate_pipeline_config(config)

        config = example_config()
        config["localization"]["ablation_tags"] = ["z", "a"]
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "sorted unique"):
            pipeline.validate_pipeline_config(config)

    def test_ambiguous_truth_initialization_has_no_legacy_alias(self):
        config = example_config()
        config["localization"]["init"] = "truth"
        config["localization"]["prior_sigma_m"] = 50.0
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "truth_position"):
            pipeline.validate_pipeline_config(config)

    def test_proposal_tuple_and_density_controls_are_authoritative(self):
        proposal = example_config()["localization"]["proposal"]
        self.assertNotIn("top_k_landmarks", proposal)
        self.assertNotIn("max_hypotheses_per_kind", proposal)
        self.assertEqual(proposal["exhaustive_tuple_limit"], 256)
        self.assertEqual(proposal["min_particles_point_fix"], 32)
        self.assertEqual(proposal["min_particles_arc"], 64)
        self.assertEqual(proposal["min_particles_single"], 128)

        for key in ("tuple_samples_per_active_solution",
                    "min_particles_point_fix", "min_particles_arc",
                    "min_particles_single"):
            with self.subTest(key=key):
                config = example_config()
                config["localization"]["proposal"][key] = 0
                with self.assertRaisesRegex(
                        build_config.InvalidConfigValue, key):
                    pipeline.validate_pipeline_config(config)

    def test_lane_names_are_path_free_identifiers(self):
        for key_path, value in (
                (("artifacts", "object_tracks_version"), "../../runs/x"),
                (("experiment", "name"), "../escape"),
                (("localization", "run_name"), "/tmp/escape")):
            with self.subTest(key_path=key_path):
                config = example_config()
                config[key_path[0]][key_path[1]] = value
                with self.assertRaisesRegex(
                        build_config.InvalidConfigValue, "path-free"):
                    pipeline.validate_pipeline_config(config)

    def test_odometry_uncertainty_and_reverse_ranges_are_strict(self):
        config = example_config()
        config["localization_inputs"]["stationary_sigma_m"] = 0.5
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "stationary_sigma_m"):
            pipeline.validate_pipeline_config(config)

        config = example_config()
        config["localization_inputs"]["reverse_keyframe_ranges"] = [
            [20, 30], [30, 40]]
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "sorted and non-overlapping"):
            pipeline.validate_pipeline_config(config)

        config = example_config()
        config["localization_inputs"]["reverse_keyframe_ranges"] = [
            [True, 10]]
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    r"must be \[start, end\] integers"):
            pipeline.validate_pipeline_config(config)

    def test_identity_review_conflicts_with_uninformative_tables(self):
        config = example_config()
        config["localization_inputs"]["identity_review_dir"] = "/reviews/r1"
        config["localization_inputs"]["use_uninformative_tables"] = True
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "identity_review_dir"):
            pipeline.validate_pipeline_config(config)

    def test_external_paths_are_resolved_without_mutating_source(self):
        config = example_config()
        original = copy.deepcopy(config)
        paths = paths_lib.FarfieldPaths(
            dataset="ds", root=Path("/farfield"),
            overrides={"dataset_base": Path("/datasets/ds")})
        resolved = pipeline.resolve_config_paths(config, paths)
        self.assertEqual(config, original)
        self.assertEqual(resolved["tracking"]["sam2_checkpoint"],
                         "/farfield/models/sam2/sam2.1_hiera_large.pt")
        self.assertEqual(
            resolved["localization_inputs"]["nominal_forward_calibration"],
            "/datasets/ds/nominal_forward.json")
        config["localization_inputs"]["identity_review_dir"] = "reviews/r1"
        resolved = pipeline.resolve_config_paths(config, paths)
        self.assertEqual(
            resolved["localization_inputs"]["identity_review_dir"],
            "/farfield/reviews/r1")


class StageOrderTest(unittest.TestCase):
    def test_stage_order_matches_artifact_dependency_graph(self):
        index = {stage: i for i, stage in enumerate(pipeline.STAGES)}
        self.assertLess(index["extract"], index["track"])
        self.assertLess(index["track"], index["audit"])
        self.assertLess(index["audit"], index["bearings"])
        self.assertLess(index["audit"], index["match"])
        self.assertLess(index["bearings"], index["localization_inputs"])
        self.assertLess(index["match"], index["localization_inputs"])
        self.assertLess(index["localization_inputs"], index["localize"])

    def test_every_artifact_stage_has_one_owner(self):
        owners = {}
        for stage, spec in pipeline.STAGE_SPECS.items():
            for kind in spec.outputs:
                self.assertNotIn(kind, owners)
                owners[kind] = stage
        self.assertEqual(set(owners),
                         (set(paths_lib.ARTIFACT_KINDS) -
                          {paths_lib.CATALOGS}) |
                         {pipeline.LOCALIZATION_RUN_KIND})


class CommandConstructionTest(unittest.TestCase):
    def setUp(self):
        self.build_identity = "a" * 64
        self.config = pipeline.resolve_config_paths(
            example_config(),
            paths_lib.FarfieldPaths(
                dataset="ds", root=Path("/farfield"),
                overrides={"dataset_base": Path("/datasets/ds")}))
        self.paths = paths_lib.FarfieldPaths(
            dataset="ds", root=Path("/farfield"),
            versions=pipeline.versions_from_config(self.config),
            overrides={"dataset_base": Path("/datasets/ds")})
        self.commands = pipeline.build_commands(
            self.paths, Path("/farfield/builds/ds/b001"), self.config,
            build_identity=self.build_identity)

    @staticmethod
    def strings(command):
        return [str(value) for value in command]

    def test_every_stage_gets_the_authoritative_build_config(self):
        self.assertEqual(set(self.commands), set(pipeline.STAGES))
        for command in self.commands.values():
            values = self.strings(command)
            self.assertIn("--build_config", values)
            self.assertIn("/farfield/builds/ds/b001/build_config.json", values)

    def test_only_localization_uses_run_dir_vocabulary(self):
        for stage, command in self.commands.items():
            values = self.strings(command)
            if stage == "localize":
                self.assertIn("--run_dir", values)
            else:
                self.assertNotIn("--run_dir", values, stage)
        localize = self.strings(self.commands["localize"])
        expected = pipeline.localization_run_dir(
            self.paths, self.config, build_identity=self.build_identity)
        self.assertIn(str(expected), localize)
        self.assertIn("--tracks-v1--build-" + "a" * 64, expected.name)

    def test_tracking_has_exactly_one_recorded_range(self):
        track = self.strings(self.commands["track"])
        self.assertEqual(track.count("--k_start"), 1)
        self.assertEqual(track.count("--k_end"), 1)
        self.assertNotIn("--range", track)
        self.assertNotIn("--skip_existing_ranges", track)

    def test_stage_inputs_and_outputs_are_explicit_artifact_dirs(self):
        audit = self.strings(self.commands["audit"])
        self.assertIn("--tracks_dir", audit)
        self.assertIn("--output_dir", audit)
        match = self.strings(self.commands["match"])
        self.assertIn("--audit_dir", match)
        self.assertIn("--catalog_dir", match)
        inputs = self.strings(self.commands["localization_inputs"])
        self.assertIn("--observations_dir", inputs)
        self.assertIn("--matching_dir", inputs)
        self.assertIn("--nominal_forward_calibration", inputs)
        sigma_index = inputs.index("--landmark_position_sigma_m")
        self.assertEqual(inputs[sigma_index + 1], "25.0")

    def test_identity_review_is_passed_only_when_recorded(self):
        inputs = self.strings(self.commands["localization_inputs"])
        self.assertNotIn("--identity_review_dir", inputs)
        config = copy.deepcopy(self.config)
        config["localization_inputs"]["identity_review_dir"] = "/reviews/r1"
        commands = pipeline.build_commands(
            self.paths, Path("/farfield/builds/ds/b001"), config,
            build_identity=self.build_identity)
        inputs = self.strings(commands["localization_inputs"])
        index = inputs.index("--identity_review_dir")
        self.assertEqual(inputs[index + 1], "/reviews/r1")

    def test_viewer_uses_exact_tracker_audit_catalog_and_output_paths(self):
        values = self.strings(pipeline.build_viewer_command(
            self.paths, self.config, build_identity=self.build_identity))
        self.assertEqual(values[:4], [
            "bazel", "run", pipeline.VIEWER_TARGET, "--"])
        expected_run = pipeline.localization_run_dir(
            self.paths, self.config, build_identity=self.build_identity)
        self.assertEqual(values[values.index("--run_dir") + 1],
                         str(expected_run))
        self.assertEqual(values[values.index("--output_dir") + 1],
                         str(expected_run) + ".viewer")
        self.assertEqual(values[values.index("--tracks_dir") + 1],
                         str(self.paths.object_tracks))
        self.assertEqual(values[values.index("--audit_dir") + 1],
                         str(self.paths.semantic_audits))
        self.assertEqual(values[values.index("--feather") + 1],
                         str(self.paths.catalogs / "catalog.feather"))

    def test_all_llm_stages_use_one_recorded_batch_staging_prefix(self):
        for stage in ("extract", "audit", "match"):
            values = self.strings(self.commands[stage])
            self.assertIn("--gcs_prefix", values)
            self.assertIn(
                "gs://REPLACE_ME/farfield/boston_harbor_leg2/b001", values)

    def test_llm_stages_have_no_incomplete_coverage_escape_hatch(self):
        for stage in ("extract", "audit", "match"):
            values = self.strings(self.commands[stage])
            self.assertNotIn("--allow_incomplete", values)
            self.assertNotIn("--allow_high_error_rate", values)

    def test_visibility_range_has_one_owner(self):
        localize = self.strings(self.commands["localize"])
        self.assertNotIn("--max_visible_range_m", localize)
        self.assertEqual(
            self.config["localization_inputs"]["max_visible_range_m"],
            15000.0)


class ManifestCompletionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = example_config()
        self.build_identity = "a" * 64
        self.paths = paths_lib.FarfieldPaths(
            dataset="ds", root=self.root,
            versions=pipeline.versions_from_config(self.config),
            overrides={"dataset_base": self.root / "datasets" / "ds"})

    def tearDown(self):
        self.temp.cleanup()

    def publish(self, kind, version, path, *, upstreams=(), config=None):
        with artifact.ArtifactDirectoryBuilder(
                path, kind=kind, dataset="ds", version=version,
                generator="pipeline_test", git_commit="test",
                arguments=(), upstreams=upstreams, config=config or {},
                declared_outputs=("payload.json",)) as builder:
            builder.output_path("payload.json").write_text("{}\n")
        return artifact.open_artifact(path)

    @staticmethod
    def shared_pinhole_config():
        return paths_lib.pinhole_manifest_config(
            {key: "0" * 64 for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS},
            resolution=24, panorama_keys=("f0000",))

    def publish_stage(self, stage, *, upstreams=None):
        if upstreams is None:
            upstreams = pipeline.expected_upstream_refs(
                self.paths, self.config, stage,
                build_identity=self.build_identity)
        refs = []
        for kind, version, path in pipeline._output_descriptors(
                self.paths, self.config, stage,
                build_identity=self.build_identity):
            output_upstreams = upstreams
            output_config = {
                "orchestration": pipeline.stage_contract(stage, self.config),
                "build_identity": self.build_identity,
            }
            if stage == "extract" and kind == paths_lib.PINHOLE_IMAGES:
                output_upstreams = ()
                output_config = self.shared_pinhole_config()
            elif stage == "extract" and kind == paths_lib.FRAME_LANDMARKS:
                output_upstreams = (refs[0],)
            refs.append(self.publish(
                kind, version, path, upstreams=output_upstreams,
                config=output_config))
        return tuple(refs)

    def publish_catalog(self):
        version = self.config["artifacts"]["catalogs_version"]
        return self.publish(paths_lib.CATALOGS, version,
                            self.paths.artifact(paths_lib.CATALOGS, version))

    def test_existence_markers_do_not_count_as_completion(self):
        build_dir = self.paths.build_dir("b001")
        build_dir.mkdir(parents=True)
        (build_dir / "track.done").write_text("done\n")
        self.assertFalse(pipeline.stage_done(
            "track", self.paths, self.config,
            build_identity=self.build_identity))

    def test_complete_manifest_is_required_and_accepted(self):
        self.publish_stage("extract")
        self.assertTrue(pipeline.stage_done(
            "extract", self.paths, self.config,
            build_identity=self.build_identity))
        self.assertFalse(pipeline.stage_done(
            "track", self.paths, self.config,
            build_identity=self.build_identity))

    def test_adopted_frame_is_complete_with_direct_result_upstream(self):
        descriptors = pipeline._output_descriptors(
            self.paths, self.config, "extract",
            build_identity=self.build_identity)
        pinhole_kind, pinhole_version, pinhole_path = descriptors[0]
        frame_kind, frame_version, frame_path = descriptors[1]
        pinhole_ref = self.publish(
            pinhole_kind, pinhole_version, pinhole_path,
            config=self.shared_pinhole_config())
        result_ref = self.publish(
            "llm_results", "adopted-results-v1",
            self.root / "adopted-results")
        self.publish(
            frame_kind, frame_version, frame_path,
            upstreams=(pinhole_ref, result_ref),
            config={
                "orchestration":
                    pipeline.stage_contract("extract", self.config),
                "build_identity": self.build_identity,
                "legacy_adoption_schema":
                    "farfield.legacy_extraction_adopted_artifact/v1",
                "legacy_adoption_report_sha256": "d" * 64,
            })

        self.assertTrue(pipeline.stage_done(
            "extract", self.paths, self.config,
            build_identity=self.build_identity))
        args = argparse.Namespace(
            build_dir=self.root / "build",
            only="extract", from_stage=None, to_stage=None,
            skip=(), dry_run=False)
        document = {
            "config": self.config,
            "build_identity": self.build_identity,
        }
        with mock.patch.object(
                pipeline, "resolve_build",
                return_value=(self.paths, document)), mock.patch.object(
                    pipeline, "build_commands",
                    return_value={"extract": ["unreachable"]}), \
                mock.patch.object(pipeline, "run") as launcher:
            pipeline.cmd_run(args, mock.Mock())
        launcher.assert_not_called()


    def test_collection_scoped_pinhole_resolves_for_extract_and_tracking(self):
        pinhole_ref, frame_ref = self.publish_stage("extract")
        manifest = artifact.load_manifest(self.paths.pinhole_images)
        self.assertEqual(dict(manifest.config), self.shared_pinhole_config())
        self.assertNotIn("orchestration", manifest.config)
        self.assertNotIn("build_identity", manifest.config)
        self.assertTrue(pipeline.stage_done(
            "extract", self.paths, self.config,
            build_identity=self.build_identity))
        self.assertEqual(
            pipeline.expected_upstream_refs(
                self.paths, self.config, "track",
                build_identity=self.build_identity),
            (pinhole_ref, frame_ref))

    def test_frame_landmarks_cannot_detach_from_configured_pinhole(self):
        self.publish_stage("extract")
        manifest_path = self.paths.frame_landmarks / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["upstreams"] = []
        artifact.atomic_write_json(manifest_path, document)

        with self.assertRaisesRegex(
                pipeline.StageContractError, "exact configured pinhole"):
            pipeline.stage_done(
                "extract", self.paths, self.config,
                build_identity=self.build_identity)
        with self.assertRaisesRegex(
                pipeline.StageDependencyError, "exact configured pinhole"):
            pipeline.expected_upstream_refs(
                self.paths, self.config, "track",
                build_identity=self.build_identity)

    def test_valid_partial_multi_output_stage_is_resumable(self):
        kind, version, path = pipeline._output_descriptors(
            self.paths, self.config, "extract",
            build_identity=self.build_identity)[0]
        self.publish(
            kind, version, path,
            config={
                "orchestration": pipeline.stage_contract("extract", self.config),
                "build_identity": self.build_identity,
            })
        self.assertFalse(pipeline.stage_done(
            "extract", self.paths, self.config,
            build_identity=self.build_identity))

    def test_extra_request_snapshot_upstream_is_allowed(self):
        self.publish_stage("extract")
        expected = pipeline.expected_upstream_refs(
            self.paths, self.config, "track",
            build_identity=self.build_identity)
        request = self.publish(
            "tracking_requests", "request-v1", self.root / "request")
        kind, version, path = pipeline._output_descriptors(
            self.paths, self.config, "track",
            build_identity=self.build_identity)[0]
        self.publish(
            kind, version, path, upstreams=(*expected, request),
            config={
                "orchestration": pipeline.stage_contract("track", self.config),
                "build_identity": self.build_identity,
            })
        self.assertTrue(pipeline.stage_done(
            "track", self.paths, self.config,
            build_identity=self.build_identity))

    def test_manifest_content_digest_is_validated(self):
        self.publish_stage("extract")
        path = self.paths.frame_landmarks / "payload.json"
        path.write_text("changed\n")
        with self.assertRaisesRegex(pipeline.StageContractError,
                                    "content digest mismatch"):
            pipeline.stage_done(
                "extract", self.paths, self.config,
                build_identity=self.build_identity)

    def test_changed_stage_config_cannot_reuse_output(self):
        self.publish_stage("extract")
        self.publish_stage("track")
        changed = copy.deepcopy(self.config)
        changed["tracking"]["window_px"] = 2048
        pipeline.validate_pipeline_config(changed)
        with self.assertRaisesRegex(pipeline.StageContractError,
                                    "different resolved configuration"):
            pipeline.stage_done(
                "track", self.paths, changed,
                build_identity=self.build_identity)

    def test_changed_upstream_identity_invalidates_descendant(self):
        self.publish_stage("extract")
        self.publish_stage("track")
        changed = copy.deepcopy(self.config)
        changed["artifacts"]["frame_landmarks_version"] = "v5"
        changed["artifacts"]["pinhole_images_version"] = "v5"
        pinhole_v5 = self.paths.artifact(paths_lib.PINHOLE_IMAGES, "v5")
        frame_v5 = self.paths.artifact(paths_lib.FRAME_LANDMARKS, "v5")
        extract_config = {
            "orchestration": pipeline.stage_contract("extract", changed),
            "build_identity": self.build_identity,
        }
        pinhole_ref = self.publish(
            paths_lib.PINHOLE_IMAGES, "v5", pinhole_v5,
            config=extract_config)
        self.publish(
            paths_lib.FRAME_LANDMARKS, "v5", frame_v5,
            upstreams=(pinhole_ref,), config=extract_config)
        with self.assertRaisesRegex(pipeline.StageContractError,
                                    "different upstream artifact identities"):
            pipeline.stage_done(
                "track", self.paths, changed,
                build_identity=self.build_identity)

    def test_stale_upstream_build_identity_cannot_start_stage(self):
        self.publish_stage("extract")
        with self.assertRaisesRegex(
                pipeline.StageDependencyError, "different immutable build"):
            pipeline.expected_upstream_refs(
                self.paths, self.config, "track",
                build_identity="b" * 64)

    def test_changed_build_identity_invalidates_completed_output(self):
        self.publish_stage("extract")
        with self.assertRaisesRegex(
                pipeline.StageContractError, "different immutable build"):
            pipeline.stage_done(
                "extract", self.paths, self.config,
                build_identity="b" * 64)

    def test_localization_run_is_itself_manifest_validated(self):
        self.publish_catalog()
        self.publish_stage("extract")
        self.publish_stage("track")
        self.publish_stage("audit")
        self.publish_stage("bearings")
        self.publish_stage("match")
        self.publish_stage("localization_inputs")
        self.publish_stage("localize")
        self.assertTrue(pipeline.stage_done(
            "localize", self.paths, self.config,
            build_identity=self.build_identity))

    def test_identity_review_is_a_post_match_typed_gate(self):
        self.config["localization_inputs"]["identity_review_dir"] = str(
            self.root / "identity_reviews" / "r1")
        self.publish_catalog()
        self.publish_stage("extract")
        self.publish_stage("track")
        self.publish_stage("audit")
        self.publish_stage("bearings")

        upstreams = pipeline.expected_upstream_refs(
            self.paths, self.config, "match",
            build_identity=self.build_identity)
        match_dir = self.paths.landmark_matches
        match_config = {
            "orchestration": pipeline.stage_contract("match", self.config),
            "build_identity": self.build_identity,
            "phase": "canonical_results",
            "coverage": "complete",
            "n_expected": 1,
            "n_successful": 1,
            "n_tracklets_expected": 1,
            "n_tracklets_successful": 1,
        }
        with artifact.ArtifactDirectoryBuilder(
                match_dir, kind=paths_lib.LANDMARK_MATCHES, dataset="ds",
                version=self.config["artifacts"][
                    "landmark_matches_version"],
                generator="pipeline_test", git_commit="test", arguments=(),
                upstreams=upstreams, config=match_config,
                declared_outputs=("matches.json",)) as builder:
            artifact.atomic_write_json(builder.output_path("matches.json"), {
                "trk-1": {"matches": [{"landmark_id": "lm-1"}]},
            })
        with self.assertRaisesRegex(
                pipeline.StageDependencyError, "identity-review gate"):
            pipeline.expected_upstream_refs(
                self.paths, self.config, "localization_inputs",
                build_identity=self.build_identity)

        matching_ref, candidates = identity_review.matching_candidates(
            match_dir)
        draft = identity_review.draft_document(matching_ref, candidates)
        draft["rows"][0].update({
            "decision": "confirmed",
            "landmark_ids": ["lm-1"],
            "reviewer": "reviewer@example.com",
            "timestamp": "2026-08-24T12:00:00+00:00",
            "notes": "checked",
        })
        input_json = self.root / "identity_review_input.json"
        artifact.atomic_write_json(input_json, draft)
        review_dir = Path(
            self.config["localization_inputs"]["identity_review_dir"])
        review_ref = identity_review.publish(
            dataset="ds", matching_dir=match_dir, input_json=input_json,
            output_dir=review_dir, version="r1")

        expected = pipeline.expected_upstream_refs(
            self.paths, self.config, "localization_inputs",
            build_identity=self.build_identity)
        self.assertEqual(expected[-1], review_ref)
        self.publish_stage("localization_inputs", upstreams=expected)
        self.assertTrue(pipeline.stage_done(
            "localization_inputs", self.paths, self.config,
            build_identity=self.build_identity))


class StageReuseProofTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset_base = testing.make_dataset(
            self.root / "datasets" / "ds", n_frames=3,
            pano_size=(32, 16))
        self.checkpoint = self.root / "models" / "sam.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"checkpoint")
        self.motion = self.dataset_base / "frames_gps.csv"
        self.calibration = self.dataset_base / "calibration.json"
        artifact.atomic_write_json(self.calibration, {
            "schema": nominal_forward.SCHEMA,
            "frame": nominal_forward.FRAME,
            "dataset": "ds",
            "version": "fixture-v1",
            "mounting_id": "fixture-rig",
            "panorama_column": 16.0,
            "panorama_width": 32,
            "bearing_camera_cw_deg": float(
                geometry.azimuth_of_pano_column(16.0, 32)) % 360.0,
            "uncertainty_deg": 1.0,
            "evidence_frame_ids": ["f0000"],
            "operator": "stage-reuse-test",
            "approved_at": "2026-08-25T12:00:00Z",
            "approved": True,
            "notes": "isolated integration fixture",
        })

        self.source_config = example_config()
        self.source_config["tracking"]["sam2_checkpoint"] = str(
            self.checkpoint)
        self.source_config["extraction"]["pinhole_resolution"] = 24
        self.source_config["tracking"]["range"] = {
            "k_start": 0, "k_end": 2}
        self.source_config["tracking"]["reference_pano_width"] = 32
        self.source_config["tracking"]["window_px"] = 32
        self.source_config["tracking"]["window_quantum"] = 8
        self.source_config["tracking"]["window_max_px"] = 32
        self.source_config["audit"]["min_supports"] = 2
        self.source_config["audit"]["max_support_chips"] = 2
        self.source_config["audit"]["max_context_chips"] = 1
        self.source_config["audit"]["chip_height_px"] = 24
        self.source_config["alignment_diagnostics"]["sun"]["n_frames"] = 3
        self.source_config["alignment_diagnostics"]["sun"][
            "min_speed_mps"] = 0.0
        self.source_config["alignment_diagnostics"]["sun"]["work_width"] = 32
        self.source_config["alignment_diagnostics"]["sweep"].update({
            "min_observations": 2,
            "min_arc_deg": 0.0,
            "max_condition": 1_000_000_000.0,
            "min_tracklets": 1,
            "min_support_frac": 0.0,
        })
        self.source_config["localization_inputs"][
            "reducer_epoch_keyframes"] = 2
        self.source_config["localization_inputs"]["motion_source"] = str(
            self.motion)
        self.source_config["localization_inputs"][
            "nominal_forward_calibration"] = str(self.calibration)
        self.source_config["localization_inputs"]["identity_review_dir"] = str(
            self.root / "identity_reviews" / "source")
        self.source_config["artifacts"]["catalogs_version"] = "catalog-old"
        self.target_config = copy.deepcopy(self.source_config)
        self.target_config["localization_inputs"]["identity_review_dir"] = None
        self.target_config["artifacts"]["catalogs_version"] = "catalog-new"
        # An explicitly downstream output version is also safe for reuse
        # through tracking.
        self.target_config["artifacts"][
            "landmark_matches_version"] = "matches-after-catalog-change"

        self.source_paths = paths_lib.FarfieldPaths(
            dataset="ds", root=self.root,
            versions=pipeline.versions_from_config(self.source_config),
            overrides={"dataset_base": self.dataset_base})
        self.target_paths = paths_lib.FarfieldPaths(
            dataset="ds", root=self.root,
            versions=pipeline.versions_from_config(self.target_config),
            overrides={"dataset_base": self.dataset_base})
        self.old_catalog = self._publish(
            paths_lib.CATALOGS, "catalog-old", self.source_paths.catalogs,
            payload=b"old catalog")
        catalog_payload = self.root / "catalog-new.feather"
        schema.build_frame(
            ids=["node:1"],
            geometries=[shapely.Point(
                testing.ANCHOR_LON + 0.001,
                testing.ANCHOR_LAT + 0.001)],
            landmark_types=["osm"],
            tags=[{"man_made": "tower", "name": "Fixture Tower"}],
        ).to_feather(catalog_payload)
        self.new_catalog = self._publish(
            paths_lib.CATALOGS, "catalog-new", self.target_paths.catalogs,
            files={"catalog.feather": catalog_payload.read_bytes()},
            config={
                "schema": schema.FULL_ARTIFACT_SCHEMA,
                "source_coverage": {
                    "schema": "farfield_catalog_source_coverage/v2",
                    "status": "passed",
                    "message": "integration fixture covers requested area",
                    "details": [],
                },
            })
        self.source_build = self.source_paths.build_dir("source")
        self.target_build = self.target_paths.build_dir("target")
        self._create_build(
            self.source_build, self.source_config, self.old_catalog,
            source_name="source-old.yaml", git_commit="source-code-commit")
        self._create_build(
            self.target_build, self.target_config, self.new_catalog,
            source_name="source-new.yaml", git_commit="target-code-commit")
        self.source_document = build_config.load(self.source_build)
        self.target_document = build_config.load(self.target_build)
        self._publish_source_prefix()
        self.checkout_patch = mock.patch.object(
            stage_reuse.provenance, "git_commit",
            return_value=self.target_document["git_commit"])
        self.checkout_patch.start()
        self.addCleanup(self.checkout_patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def _publish(self, kind, version, path, *, upstreams=(), config=None,
                 payload=b"payload", git_commit="test",
                 generator="stage_reuse_test", files=None):
        files = files or {"payload.json": payload}
        with artifact.ArtifactDirectoryBuilder(
                path, kind=kind, dataset="ds", version=version,
                generator=generator, git_commit=git_commit,
                arguments=(),
                upstreams=upstreams, config=config or {},
                declared_outputs=tuple(files)) as builder:
            for name, content in files.items():
                target = builder.output_path(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        return artifact.open_artifact(path)

    def _attestation(self):
        return stage_reuse.reviewed_prefix_code_compatibility(
            source_git_commit=self.source_document["git_commit"],
            target_git_commit=self.target_document["git_commit"],
            reviewed_by="stage-reuse-test",
            reviewed_at="2026-08-25T12:00:00+00:00",
            note="reviewed test-only prefix implementation compatibility")

    @staticmethod
    def _ref_identity(reference):
        value = reference.to_dict()
        value.pop("path")
        return value

    def _proof_document(self, source_build=None, target_build=None):
        return pipeline.stage_reuse_proof_document(
            source_build or self.source_build,
            target_build or self.target_build,
            through_stage="track",
            prefix_code_compatibility=self._attestation())

    def _create_proof(self):
        return pipeline.create_stage_reuse_proof(
            self.source_build, self.target_build, through_stage="track",
            prefix_code_compatibility=self._attestation())

    @staticmethod
    def _run_producer(module, arguments):
        previous = sys.argv
        sys.argv = [module.__name__] + [str(value) for value in arguments]
        try:
            module.main()
        finally:
            sys.argv = previous

    def _create_build(self, build_dir, config, catalog_ref, *, source_name,
                      git_commit=None, extra_inputs=None):
        if git_commit is None:
            git_commit = getattr(
                self, "target_document", {}).get(
                    "git_commit", "test-code-commit")
        source_path = self.root / source_name
        source_path.write_text(yaml.safe_dump(config))
        dataset_digests = paths_lib.dataset_source_digests(self.dataset_base)
        with mock.patch.object(
                build_config.provenance, "git_commit",
                return_value=git_commit):
            inputs = {
                "farfield_root": str(self.root.resolve()),
                "dataset_base": str(self.dataset_base.resolve()),
                "source_config": str(source_path.resolve()),
                "source_config_sha256": artifact.sha256_file(source_path),
                "sam2_checkpoint": str(self.checkpoint.resolve()),
                "sam2_checkpoint_sha256": artifact.sha256_file(
                    self.checkpoint),
                "motion_source": str(self.motion.resolve()),
                "motion_source_sha256": artifact.sha256_file(self.motion),
                "nominal_forward_calibration": str(
                    self.calibration.resolve()),
                "nominal_forward_sha256": artifact.sha256_file(
                    self.calibration),
                "catalog_manifest_digest": catalog_ref.manifest_digest,
                "catalog_content_digest": catalog_ref.content_digest,
                **dataset_digests,
            }
            review_dir = config["localization_inputs"]["identity_review_dir"]
            if review_dir is not None:
                inputs.update({
                    "identity_review_output_dir": str(Path(review_dir).resolve()),
                    "identity_review_phase": "post_match_gate",
                })
            inputs.update(extra_inputs or {})
            build_config.create(
                build_dir, dataset="ds", config=config,
                schema=pipeline.CONFIG_SCHEMA, generator="stage_reuse_test",
                inputs=inputs)

    def _publish_source_prefix(self):
        identity = self.source_document["build_identity"]
        source_digests = self.source_document["inputs"]
        context = extract_landmarks.load_artifact_validation_context(
            build_config_path=(self.source_build
                               / build_config.BUILD_CONFIG_NAME),
            dataset="ds", dataset_base=self.dataset_base)
        extraction_args = argparse.Namespace(
            dataset="ds",
            pinhole_output_dir=self.source_paths.pinhole_images,
            output_dir=self.source_paths.frame_landmarks)
        with (mock.patch.object(extract_landmarks, "NUM_WORKERS", 1),
              mock.patch.object(
                  extract_landmarks.provenance, "git_commit",
                  return_value=self.source_document["git_commit"])):
            pinhole_ref = extract_landmarks.ensure_pinhole_artifact(
                extraction_args, context, arguments=("proof-fixture",))
            request_set, request_ref, work_dir = (
                extract_landmarks.ensure_request_artifact(
                    extraction_args, context, pinhole_ref,
                    arguments=("proof-fixture",)))
            provider_landmark = {
                "primary_tag": {"key": "man_made", "value": "tower"},
                "additional_tags": [
                    {"key": "name", "value": "Fixture Tower"}],
                "confidence": "high",
                "bounding_boxes": [{
                    "yaw_angle": 0, "xmin": 250, "ymin": 200,
                    "xmax": 500, "ymax": 600,
                }],
                "description": "Fixture Tower",
            }
            results = tuple(llm_lifecycle.CanonicalResult(
                key=unit.key, attempt_id=f"proof-fixture-{index}",
                result={
                    "location_type": "harbor",
                    "landmarks": [copy.deepcopy(provider_landmark)],
                })
                            for index, unit in enumerate(request_set.units))
            result_ref = extract_landmarks.ensure_result_artifact(
                extraction_args, context, request_set, request_ref,
                results, work_dir, arguments=("proof-fixture",))
            frame_ref = extract_landmarks.publish_frame_artifact(
                extraction_args, context, pinhole_ref, request_set,
                request_ref, result_ref, results,
                arguments=("proof-fixture",))
        ingest = dataset.run_ingest(
            self.dataset_base, Path(frame_ref.path),
            dataset.IngestParams(
                fov_deg=self.source_config["ingest"]["fov_deg"],
                seam_gap_norm=self.source_config["ingest"]["seam_gap_norm"],
                seam_min_y_iou=self.source_config["ingest"][
                    "seam_min_y_iou"]))
        obs_by_frame = {
            observation.frame_idx: observation.obs_id
            for observation in ingest.observations
        }
        clean = {
            "iou": 0.8, "inter_over_mask": 0.9,
            "inter_over_box": 0.9,
        }
        track = {
            "track_id": 1,
            "birth_obs_id": obs_by_frame[0],
            "birth_keyframe": 0,
            "status": "closed",
            "close_reason": "end_of_range",
            "end_keyframe": 2,
            "last_keyframe": 2,
            "modal_label": "man_made=tower 'Fixture Tower'",
            "n_supported_keyframes": 2,
            "records": [{
                "keyframe": 0,
                "action": "birth",
                "window_origin": [0.0, 0],
                "window_px": 32,
                "health": {"ok": True},
            }] + [{
                "keyframe": keyframe,
                "action": "continue_mask",
                "window_origin": [0.0, 0],
                "window_px": 32,
                "mask_area": 64,
                "mask_bbox_window": [8, 3, 16, 10],
                "supports": [{
                    "obs_id": obs_by_frame[keyframe],
                    "class": "recorded-at-run-time",
                    "box_window": [8.0, 3.0, 16.0, 10.0],
                    **clean,
                }],
            } for keyframe in (1, 2)],
        }
        track_source_digests = {
            "build_config": artifact.sha256_file(
                self.source_build / build_config.BUILD_CONFIG_NAME),
            "dataset_tracking_inputs": artifact.sha256_json({
                key: source_digests[key]
                for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS
            }),
            "sam2_checkpoint": source_digests["sam2_checkpoint_sha256"],
            paths_lib.PINHOLE_IMAGES: pinhole_ref.content_digest,
            paths_lib.FRAME_LANDMARKS: frame_ref.content_digest,
        }
        self.track_ref = self._publish(
            paths_lib.OBJECT_TRACKS,
            self.source_config["artifacts"]["object_tracks_version"],
            self.source_paths.object_tracks,
            upstreams=(pinhole_ref, frame_ref),
            generator=stage_reuse.TRACKING_GENERATOR,
            git_commit=self.source_document["git_commit"],
            files={
                "tracks_full.json": artifact.canonical_json_bytes({
                    "range": {
                        "name": "full",
                        **self.source_config["tracking"]["range"],
                    },
                    "config": {
                        key: value for key, value in
                        self.source_config["tracking"].items()
                        if key not in {"range", "sam2_checkpoint"}
                    },
                    "tracks": [track],
                    "rejected_births": [],
                    "track_overlaps": [],
                }) + b"\n",
                "index.html": b"<html></html>\n",
            },
            config={
                "orchestration": pipeline.stage_contract(
                    "track", self.source_config),
                "schema": "farfield_object_tracks/v1",
                "coverage": "complete",
                "build_identity": identity,
                "range": {
                    "name": "full",
                    **self.source_config["tracking"]["range"],
                },
                "resolved": {
                    "ingest": self.source_config["ingest"],
                    "tracking": {
                        key: value for key, value in
                        self.source_config["tracking"].items()
                        if key != "range"
                    },
                    "gps_course": self.source_config["gps_course"],
                },
                "source_digests": track_source_digests,
            })

    def test_catalog_successor_reuses_exact_frame_and_track_refs(self):
        self.assertNotEqual(self.source_document["git_commit"],
                            self.target_document["git_commit"])
        with self.assertRaisesRegex(
                (pipeline.StageContractError, pipeline.StageDependencyError),
                "no exact stage-reuse proof"):
            pipeline.stage_done(
                "track", self.target_paths, self.target_config,
                build_identity=self.target_document["build_identity"])

        proof_path = self._create_proof()
        self.assertEqual(proof_path, self.target_build / "stage_reuse.json")
        authorization = pipeline.load_stage_reuse_proof(self.target_build)
        self.assertIsNotNone(authorization)
        self.assertIn(self.track_ref, authorization.refs)
        self.assertTrue(pipeline.stage_done(
            "extract", self.target_paths, self.target_config,
            build_identity=self.target_document["build_identity"],
            reuse_authorization=authorization))
        self.assertTrue(pipeline.stage_done(
            "track", self.target_paths, self.target_config,
            build_identity=self.target_document["build_identity"],
            reuse_authorization=authorization))

    def test_real_producers_consume_one_exact_proof_and_publish_target_lineage(
            self):
        build_path = self.target_build / build_config.BUILD_CONFIG_NAME
        execution_prefix = self.target_config["execution"][
            "batch_gcs_prefix"]
        audit_flags = [
            "--dataset", "ds",
            "--dataset_base", self.dataset_base,
            "--tracks_dir", self.target_paths.object_tracks,
            "--frame_landmarks_dir", self.target_paths.frame_landmarks,
            "--output_dir", self.target_paths.semantic_audits,
            "--build_config", build_path,
            "--orchestration_config_digest",
            pipeline.stage_contract(
                "audit", self.target_config)["config_digest"],
            "--gcs_prefix", execution_prefix,
            "--build_only",
        ]
        with self.assertRaisesRegex(SystemExit, "stage-reuse"):
            self._run_producer(audit_requests, audit_flags)

        proof_path = self._create_proof()
        proof_bytes = proof_path.read_bytes()
        proof_path.write_bytes(proof_bytes + b" ")
        with self.assertRaisesRegex(SystemExit, "stage-reuse"):
            self._run_producer(audit_requests, audit_flags)
        proof_path.write_bytes(proof_bytes)

        with mock.patch.object(
                stage_reuse.provenance, "git_commit",
                return_value="different-checkout"):
            with self.assertRaisesRegex(SystemExit, "executing checkout"):
                self._run_producer(audit_requests, audit_flags)

        self._run_producer(audit_requests, audit_flags)
        audit_work = audit_requests.audit_work_dir(
            self.target_paths.semantic_audits)
        request_dir = audit_work / "requests"
        request_set = llm_lifecycle.load_request_set(
            request_dir / llm_lifecycle.REQUEST_SET_NAME)
        self.assertEqual(len(request_set.units), 1)
        audit_payload = {
            "landmark_kind": "fixed_structure",
            "single_object": True,
            "valid_segments": [{"start_t": 0, "end_t": 2}],
            "verdict": "keep",
            "drop_reason": "none",
            "primary_object": {
                "tags": [{"tag": "man_made=tower", "weight": 1.0}],
                "name_candidates": [{
                    "name": "Fixture Tower", "weight": 1.0,
                    "basis": "reported_by_detections",
                }],
                "name_aliases": [],
                "description": "A fixed tower.",
                "distinctive_features": ["single tower"],
                "extent": "point_like",
            },
            "strike_votes": [],
            "secondary_objects": [],
            "confidence": "high",
            "unresolved": "",
        }
        audit_attempts = audit_work / audit_requests.ATTEMPTS_DIR_NAME
        for unit in request_set.units:
            llm_lifecycle.publish_attempt(
                audit_attempts,
                llm_lifecycle.Attempt(
                    request_set_fingerprint=request_set.fingerprint,
                    key=unit.key,
                    attempt_id="integration-audit",
                    response={
                        "candidates": [{"content": {"parts": [{
                            "text": json.dumps(audit_payload),
                        }]}}],
                    },
                    error=None,
                    metadata={"transport": "integration-test"},
                ))
        audit_aggregate_flags = list(audit_flags)
        audit_aggregate_flags[-1] = "--aggregate_only"
        self._run_producer(audit_requests, audit_aggregate_flags)

        bearings_flags = [
            "--dataset", "ds",
            "--dataset_base", self.dataset_base,
            "--tracks_dir", self.target_paths.object_tracks,
            "--audit_dir", self.target_paths.semantic_audits,
            "--output_dir", self.target_paths.bearing_observations,
            "--build_config", build_path,
            "--orchestration_config_digest",
            build_bearing_observations.orchestration_contract(
                self.target_document)["config_digest"],
        ]
        self._run_producer(build_bearing_observations, bearings_flags)

        matching_flags = [
            "--dataset", "ds",
            "--dataset_base", self.dataset_base,
            "--tracks_dir", self.target_paths.object_tracks,
            "--audit_dir", self.target_paths.semantic_audits,
            "--catalog_dir", self.target_paths.catalogs,
            "--output_dir", self.target_paths.landmark_matches,
            "--build_config", build_path,
            "--orchestration_config_digest",
            pipeline.stage_contract(
                "match", self.target_config)["config_digest"],
            "--gcs_prefix", execution_prefix,
            "--build_only",
        ]
        self._run_producer(match_landmarks, matching_flags)
        matching_work = match_landmarks.matching_work_dir(
            self.target_paths.landmark_matches)
        matching_requests = llm_lifecycle.load_request_set(
            matching_work / llm_lifecycle.REQUEST_SET_NAME)
        matching_attempts = (
            matching_work / match_landmarks.ATTEMPTS_DIR_NAME)
        for unit in matching_requests.units:
            entries = [{
                "set_1_id": index,
                "set_2_matches": [{
                    "set_2_id": 0,
                    "match_type": "instance",
                    "confidence": 0.9,
                }],
                "no_match_confidence": 0.1,
                "uniqueness_score": 4,
            } for index, _ in enumerate(unit.metadata["batch_keys"])]
            llm_lifecycle.publish_attempt(
                matching_attempts,
                llm_lifecycle.Attempt(
                    request_set_fingerprint=matching_requests.fingerprint,
                    key=unit.key,
                    attempt_id="integration-match",
                    response={
                        "candidates": [{"content": {"parts": [{
                            "text": json.dumps({"matches": entries}),
                        }]}}],
                    },
                    error=None,
                    metadata={"transport": "integration-test"},
                ))
        matching_aggregate_flags = [
            "--dataset", "ds",
            "--output_dir", self.target_paths.landmark_matches,
            "--aggregate_only",
        ]
        snapshot_path = matching_work / match_landmarks.WORK_SNAPSHOT_NAME
        snapshot_bytes = snapshot_path.read_bytes()
        snapshot_document = json.loads(snapshot_bytes)
        self.assertIsNotNone(snapshot_document["stage_reuse"])
        request_bytes = (
            matching_work / llm_lifecycle.REQUEST_SET_NAME).read_bytes()
        attempts_before = llm_lifecycle.load_attempts(matching_attempts)
        snapshot_document["stage_reuse"] = None
        artifact.atomic_write_json(snapshot_path, snapshot_document)
        with self.assertRaisesRegex(
                SystemExit, "immutable request-set binding"):
            self._run_producer(
                match_landmarks, matching_aggregate_flags)
        self.assertFalse(self.target_paths.landmark_matches.exists())
        self.assertEqual(
            (matching_work / llm_lifecycle.REQUEST_SET_NAME).read_bytes(),
            request_bytes)
        self.assertEqual(
            llm_lifecycle.load_attempts(matching_attempts), attempts_before)
        snapshot_path.write_bytes(snapshot_bytes)

        missing_proof = proof_path.with_name("stage_reuse.missing")
        proof_path.rename(missing_proof)
        try:
            with self.assertRaisesRegex(SystemExit, "stage reuse"):
                self._run_producer(
                    match_landmarks, matching_aggregate_flags)
        finally:
            missing_proof.rename(proof_path)
        self._run_producer(match_landmarks, matching_aggregate_flags)

        diagnostics_flags = [
            "--dataset", "ds",
            "--dataset_base", self.dataset_base,
            "--observations_dir", self.target_paths.bearing_observations,
            "--nominal_forward_calibration", self.calibration,
            "--output_dir", self.target_paths.alignment_diagnostics,
            "--build_config", build_path,
            "--orchestration_config_digest",
            build_alignment_diagnostics.orchestration_contract(
                self.target_document)["config_digest"],
        ]
        self._run_producer(build_alignment_diagnostics, diagnostics_flags)

        export_flags = [
            "--dataset", "ds",
            "--dataset_base", self.dataset_base,
            "--observations_dir", self.target_paths.bearing_observations,
            "--matching_dir", self.target_paths.landmark_matches,
            "--catalog_dir", self.target_paths.catalogs,
            "--motion_source", self.motion,
            "--nominal_forward_calibration", self.calibration,
            "--landmark_position_sigma_m",
            self.target_config["localization_inputs"][
                "landmark_position_sigma_m"],
            "--output_dir", self.target_paths.localization_inputs,
            "--build_config", build_path,
            "--orchestration_config_digest",
            build_export.orchestration_contract(
                self.target_document)["config_digest"],
        ]
        self._run_producer(build_export, export_flags)

        proof_sha = artifact.sha256_file(proof_path)
        expected_commit = self.target_document["git_commit"]
        output_paths = (
            self.target_paths.semantic_audits,
            self.target_paths.bearing_observations,
            self.target_paths.landmark_matches,
            self.target_paths.alignment_diagnostics,
            self.target_paths.localization_inputs,
        )
        for output_path in output_paths:
            manifest = artifact.load_manifest(output_path)
            self.assertEqual(manifest.git_commit, expected_commit)
            bridge = manifest.config["stage_reuse"]
            self.assertEqual(bridge["proof_sha256"], proof_sha)
            self.assertEqual(
                bridge["target_build_identity"],
                self.target_document["build_identity"])
            self.assertIn(
                self.track_ref.to_dict(), bridge["adopted_artifacts"])
        self.assertEqual(
            artifact.load_manifest(request_dir).git_commit,
            expected_commit)

    def test_executing_checkout_must_match_target_commit(self):
        with mock.patch.object(
                stage_reuse.provenance, "git_commit",
                return_value="different-checkout"):
            with self.assertRaisesRegex(
                    pipeline.StageContractError, "executing checkout"):
                self._proof_document()

    def test_same_build_early_return_reopens_the_supplied_ref(self):
        reference = self._publish(
            paths_lib.FRAME_LANDMARKS, "same-build",
            self.root / "same-build-frame",
            config={
                "build_identity": self.target_document["build_identity"]})
        supplied_manifest = artifact.load_manifest(reference.path)
        (Path(reference.path) / "payload.json").write_bytes(b"replacement")
        with self.assertRaisesRegex(
                stage_reuse.StageReuseError, "supplied frame_landmarks"):
            stage_reuse.require_compatible_artifact(
                reference, supplied_manifest,
                target_build_dir=self.target_build,
                owner_stage="extract", authorization=None)

    def test_machine_only_successor_may_keep_the_same_catalog(self):
        config = copy.deepcopy(self.target_config)
        config["artifacts"]["catalogs_version"] = "catalog-old"
        config["artifacts"]["landmark_matches_version"] = "same-catalog-match"
        paths = paths_lib.FarfieldPaths(
            dataset="ds", root=self.root,
            versions=pipeline.versions_from_config(config),
            overrides={"dataset_base": self.dataset_base})
        target = paths.build_dir("same-catalog-target")
        self._create_build(
            target, config, self.old_catalog,
            source_name="same-catalog-target.yaml",
            git_commit=self.target_document["git_commit"])
        target_document = build_config.load(target)
        attestation = stage_reuse.reviewed_prefix_code_compatibility(
            source_git_commit=self.source_document["git_commit"],
            target_git_commit=target_document["git_commit"],
            reviewed_by="stage-reuse-test",
            reviewed_at="2026-08-25T12:00:00+00:00",
            note="reviewed same-catalog machine-only successor")
        proof = pipeline.stage_reuse_proof_document(
            self.source_build, target, through_stage="track",
            prefix_code_compatibility=attestation)
        self.assertEqual(proof["source_catalog"], proof["target_catalog"])
        self.assertIn(
            "localization_inputs.identity_review_dir",
            proof["config_changed_leaves"])

    def test_attestation_must_name_the_exact_source_target_commits(self):
        invalid = stage_reuse.reviewed_prefix_code_compatibility(
            source_git_commit=self.source_document["git_commit"],
            target_git_commit="wrong-target-commit",
            reviewed_by="stage-reuse-test",
            reviewed_at="2026-08-25T12:00:00+00:00",
            note="intentionally wrong")
        with self.assertRaisesRegex(
                pipeline.StageContractError, "not bound to the source/target"):
            pipeline.stage_reuse_proof_document(
                self.source_build, self.target_build, through_stage="track",
                prefix_code_compatibility=invalid)

    def test_authorization_ref_path_is_exact_not_artifactref_equal(self):
        self._create_proof()
        authorization = pipeline.load_stage_reuse_proof(self.target_build)
        alias_ref = dataclasses.replace(
            self.track_ref, path=str(self.root / "alias" / "tracks"))
        self.assertEqual(alias_ref, self.track_ref)
        self.assertFalse(authorization.accepts(
            alias_ref, owner_stage="track",
            target_build_identity=self.target_document["build_identity"]))

    def test_direct_consumer_gets_exact_bridge_provenance(self):
        path = self._create_proof()
        authorization = pipeline.load_stage_reuse_proof(self.target_build)
        bridge = stage_reuse.require_compatible_artifact(
            self.track_ref, artifact.load_manifest(self.track_ref.path),
            target_build_dir=self.target_build, owner_stage="track",
            authorization=authorization)
        self.assertEqual(
            bridge["proof_sha256"], artifact.sha256_file(path))
        self.assertEqual(
            bridge["target_build_identity"],
            self.target_document["build_identity"])
        self.assertEqual(
            bridge["adopted_artifacts"], [self.track_ref.to_dict()])

    def test_recorded_bridge_cannot_leak_or_widen_authorization(self):
        self._create_proof()
        authorization = pipeline.load_stage_reuse_proof(self.target_build)
        track_bridge = authorization.bridge_provenance((self.track_ref,))
        stage_reuse.require_recorded_bridge(track_bridge, track_bridge)

        with self.assertRaisesRegex(
                stage_reuse.StageReuseError, "without an active"):
            stage_reuse.require_recorded_bridge(track_bridge, None)

        widened = copy.deepcopy(track_bridge)
        pinhole_ref = artifact.open_artifact(self.source_paths.pinhole_images)
        widened["adopted_artifacts"].append(pinhole_ref.to_dict())
        widened["adopted_artifacts"].sort(
            key=lambda item: (item["kind"], item["path"]))
        with self.assertRaisesRegex(
                stage_reuse.StageReuseError, "changes the authorized"):
            stage_reuse.require_recorded_bridge(widened, track_bridge)

        frame_ref = artifact.open_artifact(self.source_paths.frame_landmarks)
        audit_bridge = stage_reuse.combine_bridge_provenance(
            track_bridge, authorization.bridge_provenance((frame_ref,)))
        stage_reuse.require_recorded_bridge(
            audit_bridge, track_bridge, required_artifacts=(self.track_ref,),
            additional_artifacts=(frame_ref,))
        tampered = copy.deepcopy(audit_bridge)
        tampered["proof_sha256"] = "0" * 64
        with self.assertRaisesRegex(
                stage_reuse.StageReuseError, "different proof"):
            stage_reuse.require_recorded_bridge(
                tampered, track_bridge,
                required_artifacts=(self.track_ref,),
                additional_artifacts=(frame_ref,))

    def test_unknown_build_input_has_no_implicit_consumer(self):
        config = copy.deepcopy(self.target_config)
        config["localization"]["run_name"] = "unknown-input-target"
        paths = paths_lib.FarfieldPaths(
            dataset="ds", root=self.root,
            versions=pipeline.versions_from_config(config),
            overrides={"dataset_base": self.dataset_base})
        target = paths.build_dir("unknown-input-target")
        self._create_build(
            target, config, self.new_catalog,
            source_name="unknown-input-target.yaml",
            extra_inputs={"future_unmapped_input": "value"})
        target_document = build_config.load(target)
        attestation = stage_reuse.reviewed_prefix_code_compatibility(
            source_git_commit=self.source_document["git_commit"],
            target_git_commit=target_document["git_commit"],
            reviewed_by="stage-reuse-test",
            reviewed_at="2026-08-25T12:00:00+00:00",
            note="unknown input rejection")
        with self.assertRaisesRegex(
                pipeline.StageContractError, "unknown=.*future_unmapped_input"):
            pipeline.stage_reuse_proof_document(
                self.source_build, target, through_stage="track",
                prefix_code_compatibility=attestation)

    def test_source_track_generator_is_revalidated_not_inferred(self):
        manifest_path = self.source_paths.object_tracks / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["generator"] = "unreviewed:track_alias"
        manifest_path.write_bytes(
            artifact.canonical_json_bytes(document) + b"\n")
        with self.assertRaisesRegex(
                pipeline.StageContractError, "exact producer contract"):
            self._proof_document()

    def test_source_track_payload_shape_is_revalidated(self):
        payload_path = self.source_paths.object_tracks / "tracks_full.json"
        payload = json.loads(payload_path.read_text())
        payload["tracks"] = [{"track_id": 7, "records": []}]
        payload_path.write_bytes(artifact.canonical_json_bytes(payload) + b"\n")
        manifest_path = self.source_paths.object_tracks / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["content_digest"] = artifact.sha256_directory(
            self.source_paths.object_tracks)
        manifest_path.write_bytes(
            artifact.canonical_json_bytes(document) + b"\n")
        with self.assertRaisesRegex(
                pipeline.StageContractError, "invalid tracks"):
            self._proof_document()

    def test_proof_does_not_accept_an_unlisted_old_audit(self):
        frame_ref = artifact.open_artifact(self.source_paths.frame_landmarks)
        old_audit = self._publish(
            paths_lib.SEMANTIC_AUDITS,
            self.source_config["artifacts"]["semantic_audits_version"],
            self.source_paths.semantic_audits,
            upstreams=(self.track_ref, frame_ref),
            config={
                "orchestration": pipeline.stage_contract(
                    "audit", self.source_config),
                "build_identity": self.source_document["build_identity"],
            })
        self._create_proof()
        authorization = dataclasses.replace(
            pipeline.load_stage_reuse_proof(self.target_build),
            refs=pipeline.load_stage_reuse_proof(self.target_build).refs
                 + (old_audit,))
        # Even a contaminated ref list remains bounded by the proven prefix.
        self.assertIn(old_audit, authorization.refs)
        with self.assertRaisesRegex(
                pipeline.StageContractError, "no exact stage-reuse proof"):
            pipeline.stage_done(
                "audit", self.target_paths, self.target_config,
                build_identity=self.target_document["build_identity"],
                reuse_authorization=authorization)

    def test_tracking_change_is_not_downstream_and_is_rejected(self):
        changed_config = copy.deepcopy(self.target_config)
        changed_config["tracking"]["window_px"] += 1
        changed_paths = paths_lib.FarfieldPaths(
            dataset="ds", root=self.root,
            versions=pipeline.versions_from_config(changed_config),
            overrides={"dataset_base": self.dataset_base})
        changed_build = changed_paths.build_dir("changed")
        self._create_build(
            changed_build, changed_config, self.new_catalog,
            source_name="source-changed.yaml")
        with self.assertRaisesRegex(
                pipeline.StageContractError, "consumed by the reused prefix"):
            pipeline.stage_reuse_proof_document(
                self.source_build, changed_build, through_stage="track",
                prefix_code_compatibility=stage_reuse.
                reviewed_prefix_code_compatibility(
                    source_git_commit=self.source_document["git_commit"],
                    target_git_commit=build_config.load(changed_build)[
                        "git_commit"], reviewed_by="stage-reuse-test",
                    reviewed_at="2026-08-25T12:00:00+00:00",
                    note="reviewed test compatibility"))

    def test_output_versions_owned_through_track_cannot_change(self):
        for index, key in enumerate((
                "pinhole_images_version", "frame_landmarks_version",
                "object_tracks_version")):
            with self.subTest(key=key):
                changed_config = copy.deepcopy(self.target_config)
                changed_config["artifacts"][key] = f"changed-{index}"
                changed_paths = paths_lib.FarfieldPaths(
                    dataset="ds", root=self.root,
                    versions=pipeline.versions_from_config(changed_config),
                    overrides={"dataset_base": self.dataset_base})
                changed_build = changed_paths.build_dir(
                    f"changed-version-{index}")
                self._create_build(
                    changed_build, changed_config, self.new_catalog,
                    source_name=f"source-version-{index}.yaml")
                with self.assertRaisesRegex(
                        pipeline.StageContractError,
                        "consumed by the reused prefix"):
                    pipeline.stage_reuse_proof_document(
                        self.source_build, changed_build,
                        through_stage="track",
                        prefix_code_compatibility=stage_reuse.
                        reviewed_prefix_code_compatibility(
                            source_git_commit=self.source_document[
                                "git_commit"],
                            target_git_commit=build_config.load(changed_build)[
                                "git_commit"], reviewed_by="stage-reuse-test",
                            reviewed_at="2026-08-25T12:00:00+00:00",
                            note="reviewed test compatibility"))

    def test_build_directory_symlink_is_not_a_reuse_source(self):
        alias = self.root / "builds" / "ds" / "source-alias"
        alias.symlink_to(self.source_build, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            pipeline.stage_reuse_proof_document(
                alias, self.target_build, through_stage="track",
                prefix_code_compatibility=self._attestation())

    def test_proof_cannot_follow_a_symlink_or_survive_build_relocation(self):
        path = self._create_proof()
        actual = path.with_name("actual-proof.json")
        path.rename(actual)
        path.symlink_to(actual.name)
        with self.assertRaisesRegex(
                pipeline.StageContractError, "not a regular file"):
            pipeline.load_stage_reuse_proof(self.target_build)

        path.unlink()
        actual.rename(path)
        relocated = self.target_build.with_name("target-relocated")
        self.target_build.rename(relocated)
        with self.assertRaisesRegex(
                pipeline.StageContractError, "does not exactly reproduce"):
            pipeline.load_stage_reuse_proof(relocated)

    def test_duplicate_and_nonfinite_proof_json_are_rejected(self):
        path = self._create_proof()
        path.write_text('{"schema":"first","schema":"second"}\n')
        with self.assertRaisesRegex(
                pipeline.StageContractError, "duplicate stage-reuse JSON key"):
            pipeline.load_stage_reuse_proof(self.target_build)

        path.write_text('{"schema":NaN}\n')
        with self.assertRaisesRegex(
                pipeline.StageContractError,
                "invalid stage-reuse JSON constant"):
            pipeline.load_stage_reuse_proof(self.target_build)

    def test_semantically_equal_noncanonical_proof_is_rejected(self):
        path = self._create_proof()
        value = json.loads(path.read_text())
        path.write_text(json.dumps(value, indent=2) + "\n")
        with self.assertRaisesRegex(
                pipeline.StageContractError, "not canonical JSON"):
            pipeline.load_stage_reuse_proof(self.target_build)

    def test_semantically_equal_noncanonical_build_is_rejected(self):
        path = self.target_build / build_config.BUILD_CONFIG_NAME
        value = json.loads(path.read_text())
        path.write_text(json.dumps(value, indent=2) + "\n")
        with self.assertRaisesRegex(
                pipeline.StageContractError, "build config is not canonical"):
            self._proof_document()

    def test_proof_rejects_any_subsequent_mutation(self):
        self._create_proof()
        path = self.target_build / pipeline.STAGE_REUSE_NAME
        value = json.loads(path.read_text())
        value["compatible_artifacts"] = value["compatible_artifacts"][:-1]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(
                pipeline.StageContractError, "does not exactly reproduce"):
            pipeline.load_stage_reuse_proof(self.target_build)

    def test_proof_revalidates_target_catalog_at_every_load(self):
        self._create_proof()
        payload = self.target_paths.catalogs / "catalog.feather"
        payload.write_bytes(b"replacement catalog")
        manifest_path = self.target_paths.catalogs / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["content_digest"] = artifact.sha256_directory(
            self.target_paths.catalogs)
        manifest_path.write_bytes(
            artifact.canonical_json_bytes(document) + b"\n")
        with self.assertRaisesRegex(
                pipeline.StageContractError,
                "configured catalog differs from immutable build inputs"):
            pipeline.load_stage_reuse_proof(self.target_build)


class BuildResolutionTest(unittest.TestCase):
    def test_build_directory_resolves_root_dataset_and_all_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = example_config()
            build_dir = root / "builds" / "ds" / "b001"
            dataset_base = root / "datasets" / "ds"
            build_config.create(
                build_dir, dataset="ds", config=config,
                schema=pipeline.CONFIG_SCHEMA, generator="test",
                inputs={"farfield_root": root,
                        "dataset_base": dataset_base})
            paths, document = pipeline.resolve_build(build_dir)
            self.assertEqual(paths.root, root)
            self.assertEqual(paths.dataset_base, dataset_base)
            self.assertEqual(paths.versions,
                             pipeline.versions_from_config(config))
            self.assertEqual(document["dataset"], "ds")

    def test_build_config_cannot_be_relocated_to_another_vocabulary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = example_config()
            wrong = root / "runs" / "ds" / "b001"
            build_config.create(
                wrong, dataset="ds", config=config,
                schema=pipeline.CONFIG_SCHEMA, generator="test",
                inputs={"farfield_root": root,
                        "dataset_base": root / "datasets" / "ds"})
            with self.assertRaisesRegex(ValueError, "build config says"):
                pipeline.resolve_build(wrong)


class ParserContractTest(unittest.TestCase):
    def test_run_has_no_result_shaping_or_mutation_flags(self):
        args = pipeline.build_parser().parse_args(
            ["run", "--build_dir", "/farfield/builds/ds/b001"])
        self.assertFalse(hasattr(args, "range"))
        self.assertFalse(hasattr(args, "force"))
        self.assertFalse(hasattr(args, "uninformative_tables"))
        self.assertEqual(args.build_dir, Path("/farfield/builds/ds/b001"))


if __name__ == "__main__":
    unittest.main()
