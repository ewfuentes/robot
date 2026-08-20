import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import run_config


CONFIG = {
    "extraction": {"model": "gemini-3.1-pro-preview", "resolution": 2048},
    "tracking": {"epoch_keyframes": 5, "min_supports": 2},
    "catalog": "v3_trimmed",
}

REQUIRED = ("extraction.model", "extraction.resolution",
            "tracking.epoch_keyframes", "tracking.min_supports", "catalog")


class CreateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "r001"

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_validates_all_required_keys_at_once(self):
        incomplete = {"extraction": {"model": "m"}}
        with self.assertRaises(run_config.MissingConfigValue) as ctx:
            run_config.create(self.run_dir, incomplete, required=REQUIRED,
                              generator="g", inputs={})
        message = str(ctx.exception)
        for key in ("extraction.resolution", "tracking.epoch_keyframes",
                    "catalog"):
            self.assertIn(key, message)
        self.assertFalse(self.run_dir.exists())  # nothing written on failure

    def test_create_then_load_round_trips(self):
        run_config.create(self.run_dir, CONFIG, required=REQUIRED,
                          generator="farfield.pipeline",
                          inputs={"dataset_base": "/data/x"})
        doc = run_config.load(self.run_dir)
        self.assertEqual(doc["config"], CONFIG)
        self.assertEqual(doc["inputs"]["dataset_base"], "/data/x")
        self.assertTrue(doc["git_commit"])

    def test_runs_are_immutable(self):
        run_config.create(self.run_dir, CONFIG, required=REQUIRED,
                          generator="g", inputs={})
        with self.assertRaises(FileExistsError):
            run_config.create(self.run_dir, CONFIG, required=REQUIRED,
                              generator="g", inputs={})


class ValueTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "r001"
        run_config.create(self.run_dir, CONFIG, required=REQUIRED,
                          generator="g", inputs={})

    def tearDown(self):
        self.tmp.cleanup()

    def test_dotted_access(self):
        self.assertEqual(
            run_config.value(self.run_dir, "tracking.epoch_keyframes"), 5)
        self.assertEqual(run_config.value(self.run_dir, "catalog"),
                         "v3_trimmed")

    def test_missing_value_names_the_run_and_key(self):
        with self.assertRaises(run_config.MissingConfigValue) as ctx:
            run_config.value(self.run_dir, "tracking.does_not_exist")
        self.assertIn("tracking.does_not_exist", str(ctx.exception))
        self.assertIn("run_config.json", str(ctx.exception))

    def test_not_a_run_dir_is_a_pointed_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            run_config.load(Path(self.tmp.name) / "not_a_run")
        self.assertIn("not a run directory", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
