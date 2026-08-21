import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from experimental.overhead_matching.swag.farfield import (
    paths as paths_lib,
    pipeline,
    run_config,
)

CONFIG_PATH = Path(__file__).parent / "configs" / "harbor_example.yaml"


def example_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


class ConfigContractTest(unittest.TestCase):
    def test_example_config_satisfies_every_required_key(self):
        """The checked-in example must never drift behind REQUIRED_CONFIG."""
        config = example_config()
        missing = [key for key in pipeline.REQUIRED_CONFIG
                   if run_config._get(config, key) is None]
        self.assertEqual(missing, [])

    def test_missing_key_is_refused_at_run_creation(self):
        config = example_config()
        del config["fusion"]["epoch_keyframes"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(run_config.MissingConfigValue) as ctx:
                run_config.create(Path(tmp) / "r001", config,
                                  required=pipeline.REQUIRED_CONFIG,
                                  generator="test", inputs={})
            self.assertIn("fusion.epoch_keyframes", str(ctx.exception))


class StageOrderTest(unittest.TestCase):
    """The ordering the docs used to carry in prose, pinned in code."""

    def test_every_stage_is_declared_once(self):
        self.assertEqual(len(set(pipeline.STAGES)), len(pipeline.STAGES))

    def test_dependency_order(self):
        idx = {stage: i for i, stage in enumerate(pipeline.STAGES)}
        self.assertLess(idx["extract"], idx["track"])
        self.assertLess(idx["track"], idx["audit"])
        self.assertLess(idx["audit"], idx["offset"])   # tracklets need audits
        self.assertLess(idx["audit"], idx["match"])
        self.assertLess(idx["offset"], idx["export"])  # export bakes offset
        self.assertLess(idx["match"], idx["export"])   # export takes tables
        self.assertLess(idx["export"], idx["localize"])
        self.assertLess(idx["localize"], idx["plots"])
        self.assertLess(idx["localize"], idx["viewer"])

    def test_detection_consumers_all_follow_extract(self):
        idx = {stage: i for i, stage in enumerate(pipeline.STAGES)}
        for stage in pipeline.DETECTION_CONSUMERS:
            self.assertGreater(idx[stage], idx["extract"])

    def test_every_stage_has_a_completion_marker(self):
        paths = paths_lib.FarfieldPaths(
            dataset="x", root=Path("/nonexistent"),
            versions={paths_lib.FRAME_LANDMARKS: "v1",
                      paths_lib.OBJECT_TRACKS: "v1"})
        run_dir = Path("/nonexistent/run")
        loc_run = Path("/nonexistent/loc")
        for stage in pipeline.STAGES:
            if stage == "track":
                continue  # its marker probe imports torch; covered elsewhere
            self.assertFalse(pipeline.stage_done(stage, paths, run_dir,
                                                 loc_run), stage)


class CommandConstructionTest(unittest.TestCase):
    """Every stage command is constructible from a recorded config alone —
    no value invented by the orchestrator."""

    def test_commands_carry_only_recorded_values(self):
        config = example_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = (root / "artifacts" / "object_tracks" / "ds" / "v1"
                       / "runs" / "r001")
            run_dir.mkdir(parents=True)
            paths = paths_lib.FarfieldPaths(
                dataset="ds", root=root,
                versions={paths_lib.FRAME_LANDMARKS: "v4",
                          paths_lib.PINHOLE_IMAGES: "v1",
                          paths_lib.OBJECT_TRACKS: "v1"},
                catalog="v3_trimmed")
            args = SimpleNamespace(approve_cost=False, online=False,
                                   uninformative_tables=False, range=None,
                                   notes="")
            commands = pipeline.build_commands(
                paths, run_dir, config, root / "runs" / "exp" / "ds_r001",
                args)
        self.assertEqual(set(commands), set(pipeline.STAGES))
        extract = [str(c) for c in commands["extract"]]
        self.assertIn("gemini-3.1-pro-preview", extract)
        self.assertIn("osm_tags_farfield", extract)
        self.assertNotIn("--force", extract)  # never implied (re-bills)
        match = [str(c) for c in commands["match"]]
        self.assertIn("v3_trimmed", match)
        export = [str(c) for c in commands["export"]]
        self.assertIn("15000.0", export)
        localize = [str(c) for c in commands["localize"]]
        self.assertIn("uniform", localize)
        self.assertIn("15000.0", localize)  # same recorded radius everywhere
        track = [str(c) for c in commands["track"]]
        self.assertIn("--skip_existing_ranges", track)

    def test_uninformative_tables_flag_swaps_the_export_source(self):
        config = example_config()
        paths = paths_lib.FarfieldPaths(dataset="ds", root=Path("/r"),
                                        catalog="v3_trimmed")
        run_dir = Path("/r/artifacts/object_tracks/ds/v1/runs/r001")
        for flag, expected in ((True, "uninformative"),
                               (False, str(run_dir / "matching"
                                           / "compatibility.json"))):
            args = SimpleNamespace(approve_cost=False, online=False,
                                   uninformative_tables=flag, range=None,
                                   notes="")
            commands = pipeline.build_commands(paths, run_dir, config,
                                               Path("/r/runs/e/x"), args)
            export = [str(c) for c in commands["export"]]
            self.assertIn(expected, export)


class ExtractionGateTest(unittest.TestCase):
    def test_incomplete_extraction_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            fl = Path(tmp) / "fl"
            fl.mkdir()
            (fl / "manifest.json").write_text(json.dumps(
                {"config": {"complete": False, "n_no_usable_response": 7}}))
            paths = SimpleNamespace(frame_landmarks=fl)
            with self.assertRaises(SystemExit) as ctx:
                pipeline.check_extraction(paths, dry_run=False)
            self.assertIn("retry_failed", str(ctx.exception))

    def test_missing_manifest_stops(self):
        paths = SimpleNamespace(frame_landmarks=Path("/nonexistent"))
        with self.assertRaises(SystemExit):
            pipeline.check_extraction(paths, dry_run=False)

    def test_complete_extraction_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fl = Path(tmp) / "fl"
            fl.mkdir()
            (fl / "manifest.json").write_text(json.dumps(
                {"config": {"complete": True}}))
            pipeline.check_extraction(SimpleNamespace(frame_landmarks=fl),
                                      dry_run=False)


if __name__ == "__main__":
    unittest.main()
