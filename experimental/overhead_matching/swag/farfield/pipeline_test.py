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
    artifact_recipe,
    build_config,
    dataset,
    geometry,
    llm_lifecycle,
    nominal_forward,
    paths as paths_lib,
    pipeline,
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

    def test_viewer_settings_come_from_the_recorded_config(self):
        """The command and the completeness check must read one source.

        A literal expectation in the orchestrator plus a default in the viewer
        is two owners of the same value: change either and every already-built
        viewer is declared stale by a change nobody made.
        """
        config = copy.deepcopy(self.config)
        config["viewer"] = {"max_particles": 1234, "basemap_detail": 4,
                            "embed_source_chips": False}
        values = self.strings(pipeline.build_viewer_command(
            self.paths, config, build_identity=self.build_identity))
        self.assertEqual(values[values.index("--max_particles") + 1], "1234")
        self.assertEqual(values[values.index("--basemap_detail") + 1], "4.0")
        self.assertIn("--no_source_chips", values)
        # What the viewer will record, and what the check will demand, are the
        # same object -- including the int-vs-float spelling the flag parses.
        self.assertEqual(pipeline.viewer_config(config), {
            "max_particles": 1234, "basemap_detail": 4.0,
            "body_only": False, "embed_source_chips": False})


    def test_viewer_omits_the_chip_flag_when_chips_are_enabled(self):
        values = self.strings(pipeline.build_viewer_command(
            self.paths, self.config, build_identity=self.build_identity))
        self.assertNotIn("--no_source_chips", values)
        self.assertTrue(
            pipeline.viewer_config(self.config)["embed_source_chips"])

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

    def publish(self, kind, version, path, *, upstreams=(), config=None,
                artifact_identity=None, recipe=None):
        with artifact.ArtifactDirectoryBuilder(
                path, kind=kind, dataset="ds", version=version,
                generator="pipeline_test", git_commit="test",
                arguments=(), upstreams=upstreams, config=config or {},
                artifact_identity=artifact_identity, recipe=recipe,
                declared_outputs=("payload.json",)) as builder:
            builder.output_path("payload.json").write_text("{}\n")
        return artifact.open_artifact(path)

    @staticmethod
    def shared_pinhole_config():
        return paths_lib.pinhole_manifest_config(
            {key: "0" * 64 for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS},
            resolution=24, panorama_keys=("f0000",))

    def publish_stage(self, stage, *, upstreams=None, build_inputs=None):
        if upstreams is None:
            upstreams = pipeline.expected_upstream_refs(
                self.paths, self.config, stage)
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
            identity, recipe = None, None
            if build_inputs is not None and kind in \
                    pipeline.PIPELINE_ARTIFACT_OWNER:
                identity = pipeline.expected_artifact_identity(
                    self.paths, self.config, kind, build_inputs=build_inputs)
                recipe = pipeline.stage_recipe(
                    self.paths, self.config,
                    pipeline.PIPELINE_ARTIFACT_OWNER[kind],
                    build_inputs=build_inputs)
            refs.append(self.publish(
                kind, version, path, upstreams=output_upstreams,
                config=output_config, artifact_identity=identity,
                recipe=recipe))
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
                self.paths, self.config, "track"),
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
                self.paths, self.config, "track")

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
            self.paths, self.config, "track")
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

    def test_a_different_build_identity_alone_does_not_invalidate(self):
        """The point of the change. A whole-build digest moved whenever ANY
        config leaf moved, so retuning the filter invalidated paid extraction
        and `stage_reuse` had to grant human-attested exceptions. Identity is
        data lineage now, and a build identity is a label."""
        self.publish_stage("extract")
        self.assertEqual(
            pipeline.expected_upstream_refs(
                self.paths, self.config, "track"),
            pipeline.expected_upstream_refs(
                self.paths, self.config, "track"))
        self.assertTrue(pipeline.stage_done(
            "extract", self.paths, self.config, build_identity="b" * 64))

    def test_code_lineage_reads_the_published_provenance(self):
        """The readout for `code_provenance`, which nothing else consumes.

        Its previous reader looked in `manifest.config` for a block that is a
        top-level manifest field, so it reported "no artifacts in this
        lineage" for every corpus -- indistinguishable from a corpus with
        nothing stamped, which is why the bug survived a run against real
        data. Pin both directions."""
        empty = pipeline.code_lineage(
            self.paths, self.config, build_identity=self.build_identity)
        self.assertIn("no artifacts", empty)
        self.publish_stage("extract")
        described = pipeline.code_lineage(
            self.paths, self.config, build_identity=self.build_identity)
        self.assertNotIn("no artifacts", described)

    def test_a_stage_published_with_its_identity_passes_the_gate(self):
        """What the pipeline actually does. Every other test here calls
        `stage_done` WITHOUT build_inputs, which skips the identity branch
        entirely -- so for a while nothing recorded an identity at all and no
        test could notice, while `cmd_run` and `cmd_status` always pass
        build_inputs and would have rejected every freshly built stage."""
        inputs = {"dataset_panorama_sha256": "c" * 64}
        self.publish_stage("extract", build_inputs=inputs)
        manifest = artifact.load_manifest(self.paths.frame_landmarks)
        self.assertEqual(
            manifest.artifact_identity,
            pipeline.expected_artifact_identity(
                self.paths, self.config, paths_lib.FRAME_LANDMARKS,
                build_inputs=inputs))
        self.assertTrue(pipeline.stage_done(
            "extract", self.paths, self.config,
            build_identity=self.build_identity, build_inputs=inputs))

    def test_a_stage_published_without_an_identity_is_refused(self):
        """The honest outcome for a producer driven by hand."""
        self.publish_stage("extract")
        with self.assertRaisesRegex(
                pipeline.StageContractError, "records no identity"):
            pipeline.stage_done(
                "extract", self.paths, self.config,
                build_identity=self.build_identity,
                build_inputs={"dataset_panorama_sha256": "c" * 64})

    def test_every_stage_has_at_most_one_gated_output(self):
        """`--artifact_identity` is one flag, so this must stay true."""
        for stage in pipeline.STAGES:
            gated = [kind for kind in pipeline.STAGE_SPECS[stage].outputs
                     if kind in pipeline.PIPELINE_ARTIFACT_OWNER]
            self.assertLessEqual(len(gated), 1, stage)

    def test_the_flag_is_omitted_for_a_stage_with_no_gated_output(self):
        self.assertEqual(
            pipeline.stage_identity_flags(
                self.paths, self.config, "localize",
                build_inputs={"dataset_panorama_sha256": "c" * 64}),
            [])

    def test_a_published_stage_is_self_describing(self):
        """Requirement: an artifact says how to reproduce it, without a join.

        Both terms of its identity that a manifest cannot otherwise recover --
        the resolved stage config and the build inputs the stage read -- are
        recorded, so the identity recomputes from the manifest alone. Before
        this, answering either question meant joining through `build_identity`
        to `builds/`, which the docs call orchestration state and nothing
        protects."""
        inputs = {"dataset_panorama_sha256": "c" * 64}
        self.publish_stage("extract", build_inputs=inputs)
        manifest = artifact.load_manifest(self.paths.frame_landmarks)
        artifact_recipe.verify_self_describing(manifest)
        self.assertEqual(
            artifact_recipe.identity_from_manifest(manifest),
            pipeline.expected_artifact_identity(
                self.paths, self.config, paths_lib.FRAME_LANDMARKS,
                build_inputs=inputs))

    def test_the_recorded_stage_config_reproduces_the_contract_digest(self):
        """The recipe stores exactly what `config_digest` hashes -- storing
        anything else would record a config that cannot reproduce its own
        digest."""
        inputs = {"dataset_panorama_sha256": "c" * 64}
        self.publish_stage("extract", build_inputs=inputs)
        manifest = artifact.load_manifest(self.paths.frame_landmarks)
        self.assertEqual(
            artifact_recipe.stage_config_digest(manifest.recipe),
            pipeline.stage_contract("extract", self.config)["config_digest"])

    def test_every_stage_publishes_a_self_describing_artifact(self):
        """The end-to-end property, walked in the order a real run walks it.

        Each stage's recipe is built from upstreams that exist because the
        previous stage published them, exactly as `cmd_run` does -- so this
        also proves `stage_recipe` can be built at the moment the pipeline
        needs it, not just in principle."""
        inputs = {"dataset_panorama_sha256": "c" * 64}
        self.publish_catalog()
        for stage in pipeline.STAGES:
            gated = [kind for kind in pipeline.STAGE_SPECS[stage].outputs
                     if kind in pipeline.PIPELINE_ARTIFACT_OWNER]
            if not gated:
                continue
            with self.subTest(stage=stage):
                self.publish_stage(stage, build_inputs=inputs)
                version = pipeline._value(
                    self.config, pipeline.VERSION_KEYS[gated[0]])
                manifest = artifact.load_manifest(
                    self.paths.artifact(gated[0], version))
                artifact_recipe.verify_self_describing(manifest)
                self.assertEqual(
                    artifact_recipe.stage_config_digest(manifest.recipe),
                    pipeline.stage_contract(
                        stage, self.config)["config_digest"])

    def test_a_changed_input_digest_does_invalidate(self):
        """What replaces it. The checkpoint's PATH is in the stage config but
        its BYTES are only in the build inputs, so this is the term that
        stops tracks being reused against different weights."""
        self.publish_stage("extract")
        inputs = {"dataset_panorama_sha256": "e" * 64}
        self.assertNotEqual(
            pipeline.expected_artifact_identity(
                self.paths, self.config, paths_lib.FRAME_LANDMARKS,
                build_inputs=inputs),
            pipeline.expected_artifact_identity(
                self.paths, self.config, paths_lib.FRAME_LANDMARKS,
                build_inputs={"dataset_panorama_sha256": "f" * 64}))

    def test_post_track_outputs_use_stage_scoped_identity(self):
        self.publish_catalog()
        self.publish_stage("extract")
        self.publish_stage("track")
        self.publish_stage("audit")
        self.publish_stage("bearings")
        self.publish_stage("match")
        inputs_ref, = self.publish_stage("localization_inputs")

        successor_identity = "b" * 64
        self.assertTrue(pipeline.stage_done(
            "localization_inputs", self.paths, self.config,
            build_identity=successor_identity))
        self.assertEqual(
            pipeline.expected_upstream_refs(
                self.paths, self.config, "localize"),
            (inputs_ref,))

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
            self.paths, self.config, "match")
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
                self.paths, self.config, "localization_inputs")

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
            self.paths, self.config, "localization_inputs")
        self.assertEqual(expected[-1], review_ref)
        self.publish_stage("localization_inputs", upstreams=expected)
        self.assertTrue(pipeline.stage_done(
            "localization_inputs", self.paths, self.config,
            build_identity=self.build_identity))


class BuildResolutionTest(unittest.TestCase):
    def test_build_directory_resolves_root_dataset_and_all_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = example_config()
            build_dir = root / "builds" / "ds" / "b001"
            dataset_base = root / "datasets" / "ds"
            build_config.create(
                build_dir, dataset="ds", config=config,
                **pipeline.SCHEMA_ARGS, generator="test",
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
                **pipeline.SCHEMA_ARGS, generator="test",
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
