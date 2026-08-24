import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    paths as paths_lib,
    pipeline,
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


class ConfigContractTest(unittest.TestCase):
    def test_example_is_exact_and_fully_resolved(self):
        config = example_config()
        pipeline.validate_pipeline_config(config)
        self.assertEqual(config["tracking"]["range"],
                         {"k_start": 0, "k_end": 235})
        self.assertEqual(
            config["localization_inputs"]["landmark_position_sigma_m"],
            25.0)

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
            self.paths, Path("/farfield/builds/ds/b001"), self.config)

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
        self.assertIn("/farfield/runs/260821_example_experiment/"
                      "boston_harbor_leg2_v1", localize)

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

    def publish_stage(self, stage, *, upstreams=None):
        if upstreams is None:
            upstreams = pipeline.expected_upstream_refs(
                self.paths, self.config, stage,
                build_identity=self.build_identity)
        refs = []
        for kind, version, path in pipeline._output_descriptors(
                self.paths, self.config, stage):
            refs.append(self.publish(
                kind, version, path, upstreams=upstreams,
                config={
                    "orchestration": pipeline.stage_contract(stage, self.config),
                    "build_identity": self.build_identity,
                }))
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

    def test_valid_partial_multi_output_stage_is_resumable(self):
        kind, version, path = pipeline._output_descriptors(
            self.paths, self.config, "extract")[0]
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
            self.paths, self.config, "track")[0]
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
