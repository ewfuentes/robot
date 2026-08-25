import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import build_config


CONFIG = {
    "artifacts": {"object_tracks_version": "v1"},
    "tracking": {"range": {"k_start": 0, "k_end": 9}},
}
REQUIRED = ("artifacts.object_tracks_version", "tracking.range.k_start",
            "tracking.range.k_end")
SCHEMA = {
    "artifacts.object_tracks_version": build_config.ValueSpec(
        (str,), nonempty=True),
    "tracking.range.k_start": build_config.ValueSpec((int,), minimum=0),
    "tracking.range.k_end": build_config.ValueSpec((int,), minimum=0),
}


class BuildConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.build_dir = Path(self.tmp.name) / "b001"

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_and_identity(self):
        build_config.create(
            self.build_dir, dataset="ds", config=CONFIG, schema=SCHEMA,
            generator="test", inputs={"dataset_base": "/data/ds"})
        document = build_config.load(self.build_dir)
        self.assertEqual(document["dataset"], "ds")
        self.assertEqual(document["config"], CONFIG)
        self.assertEqual(len(document["build_identity"]), 64)

    def test_reports_all_missing_values_before_writing(self):
        with self.assertRaises(build_config.MissingConfigValue) as ctx:
            build_config.create(
                self.build_dir, dataset="ds", config={}, required=REQUIRED,
                generator="test", inputs={})
        self.assertIn("tracking.range.k_end", str(ctx.exception))
        self.assertFalse(self.build_dir.exists())

    def test_refuses_nonempty_workspace(self):
        self.build_dir.mkdir(parents=True)
        (self.build_dir / "stale").write_text("x")
        with self.assertRaises(FileExistsError):
            build_config.create(
                self.build_dir, dataset="ds", config=CONFIG,
                required=REQUIRED, generator="test", inputs={})

    def test_refuses_symlinked_build_directory_without_writing_target(self):
        target = Path(self.tmp.name) / "target"
        target.mkdir()
        self.build_dir.symlink_to(target, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            build_config.create(
                self.build_dir, dataset="ds", config=CONFIG,
                required=REQUIRED, generator="test", inputs={})
        self.assertEqual(list(target.iterdir()), [])

    def test_load_rejects_symlinked_recipe(self):
        build_config.create(
            self.build_dir, dataset="ds", config=CONFIG, required=REQUIRED,
            generator="test", inputs={})
        recipe = self.build_dir / build_config.BUILD_CONFIG_NAME
        target = self.build_dir / "actual.json"
        recipe.rename(target)
        recipe.symlink_to(target.name)
        with self.assertRaisesRegex(FileNotFoundError, "non-symlink"):
            build_config.load(self.build_dir)

    def test_load_rejects_duplicate_json_keys(self):
        self.build_dir.mkdir()
        path = self.build_dir / build_config.BUILD_CONFIG_NAME
        path.write_text('{"schema":"first","schema":"second"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            build_config.load(self.build_dir)

    def test_detects_tampering(self):
        build_config.create(
            self.build_dir, dataset="ds", config=CONFIG, required=REQUIRED,
            generator="test", inputs={})
        path = self.build_dir / build_config.BUILD_CONFIG_NAME
        document = json.loads(path.read_text())
        document["config"]["tracking"]["range"]["k_end"] = 99
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            build_config.load(self.build_dir)

    def test_value_never_defaults(self):
        build_config.create(
            self.build_dir, dataset="ds", config=CONFIG, required=REQUIRED,
            generator="test", inputs={})
        self.assertEqual(
            build_config.value(self.build_dir, "tracking.range.k_end"), 9)
        with self.assertRaises(build_config.MissingConfigValue):
            build_config.value(self.build_dir, "tracking.missing")

    def test_schema_rejects_unknown_leaf(self):
        config = json.loads(json.dumps(CONFIG))
        config["tracking"]["ranges"] = []
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "tracking.ranges"):
            build_config.create(
                self.build_dir, dataset="ds", config=config, schema=SCHEMA,
                generator="test", inputs={})

    def test_schema_rejects_bool_as_integer(self):
        config = json.loads(json.dumps(CONFIG))
        config["tracking"]["range"]["k_start"] = True
        with self.assertRaisesRegex(build_config.InvalidConfigValue,
                                    "must be int"):
            build_config.create(
                self.build_dir, dataset="ds", config=config, schema=SCHEMA,
                generator="test", inputs={})

    def test_schema_rejects_out_of_range_value(self):
        config = json.loads(json.dumps(CONFIG))
        config["tracking"]["range"]["k_end"] = -1
        with self.assertRaisesRegex(build_config.InvalidConfigValue, ">= 0"):
            build_config.create(
                self.build_dir, dataset="ds", config=config, schema=SCHEMA,
                generator="test", inputs={})


if __name__ == "__main__":
    unittest.main()
