import argparse
import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import paths as paths_lib


def make_parser(**kwargs) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    paths_lib.add_arguments(parser, **kwargs)
    return parser


class ResolutionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "datasets" / "boston_harbor_leg2").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def paths_for(self, argv, parser=None, **from_args_kwargs):
        parser = parser or make_parser(feather=True, video=True,
                                       checkpoint=True, pinhole=True)
        args = parser.parse_args(argv + ["--farfield_root", str(self.root)])
        return paths_lib.from_args(args, **from_args_kwargs)

    def test_dataset_lane_resolves_from_name(self):
        p = self.paths_for(["--dataset", "boston_harbor_leg2"])
        self.assertEqual(p.dataset_base,
                         self.root / "datasets" / "boston_harbor_leg2")
        self.assertEqual(p.panorama_dir, p.dataset_base / "panorama")

    def test_no_dataset_anywhere_is_an_error(self):
        with self.assertRaises(paths_lib.MissingInput):
            self.paths_for([])

    def test_no_default_artifact_version(self):
        p = self.paths_for(["--dataset", "boston_harbor_leg2"])
        with self.assertRaises(paths_lib.MissingInput):
            _ = p.frame_landmarks
        p2 = self.paths_for(["--dataset", "boston_harbor_leg2",
                             "--frame_landmarks_version", "v4"])
        self.assertEqual(
            p2.frame_landmarks,
            self.root / "artifacts" / "frame_landmarks"
            / "boston_harbor_leg2" / "v4")

    def test_no_default_catalog(self):
        p = self.paths_for(["--dataset", "boston_harbor_leg2"])
        with self.assertRaises(paths_lib.MissingInput):
            _ = p.feather
        p2 = self.paths_for(["--dataset", "boston_harbor_leg2",
                             "--catalog", "v3_trimmed"])
        self.assertEqual(
            p2.feather,
            self.root / "artifacts" / "catalogs" / "boston_harbor_leg2"
            / "v3_trimmed.feather")

    def test_no_default_sam2_checkpoint(self):
        p = self.paths_for(["--dataset", "boston_harbor_leg2"])
        with self.assertRaises(paths_lib.MissingInput):
            _ = p.sam2_checkpoint
        p2 = self.paths_for(["--dataset", "boston_harbor_leg2",
                             "--checkpoint", "/x/ckpt.pt"])
        self.assertEqual(p2.sam2_checkpoint, Path("/x/ckpt.pt"))

    def test_explicit_flags_override_resolution(self):
        p = self.paths_for(["--dataset", "boston_harbor_leg2",
                            "--feather", "/elsewhere/cat.feather",
                            "--landmark_base", "/elsewhere/fl"])
        self.assertEqual(p.feather, Path("/elsewhere/cat.feather"))
        self.assertEqual(p.frame_landmarks, Path("/elsewhere/fl"))

    def test_dataset_base_alone_names_the_dataset(self):
        p = self.paths_for(["--dataset_base",
                            str(self.root / "datasets" / "boston_harbor_leg2")])
        self.assertEqual(p.dataset, "boston_harbor_leg2")

    def test_video_requires_metadata_entry(self):
        meta = self.root / "datasets" / "boston_harbor_leg2" / \
            "pipeline_metadata.json"
        meta.write_text(json.dumps({"video": {}}))
        p = self.paths_for(["--dataset", "boston_harbor_leg2"])
        with self.assertRaises(paths_lib.MissingInput):
            _ = p.video

    def test_video_strips_retention_note_and_roots_relative_paths(self):
        meta = self.root / "datasets" / "boston_harbor_leg2" / \
            "pipeline_metadata.json"
        meta.write_text(json.dumps({"video": {
            "source_video": "raw_material/x/leg2.mp4 (not retained)"}}))
        p = self.paths_for(["--dataset", "boston_harbor_leg2"])
        self.assertEqual(p.video, self.root / "raw_material" / "x" /
                         "leg2.mp4")

    def test_require_lists_all_missing_at_once(self):
        p = self.paths_for(["--dataset", "boston_harbor_leg2"])
        with self.assertRaises(paths_lib.MissingInput) as ctx:
            p.require("panorama_dir", "feather")
        self.assertIn("panorama_dir", str(ctx.exception))
        self.assertIn("feather", str(ctx.exception))

    def test_describe_tolerates_unresolved_inputs(self):
        p = self.paths_for(["--dataset", "boston_harbor_leg2"])
        text = p.describe()
        self.assertIn("<unresolved>", text)
        self.assertIn("boston_harbor_leg2", text)


class InferFromArtifactPathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run_dir = (self.root / "artifacts" / "object_tracks"
                        / "pohang_canal_04" / "v1" / "runs" / "r002")
        self.run_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_infers_dataset_root_and_version(self):
        inferred = paths_lib.infer_from_artifact_path(self.run_dir)
        self.assertEqual(inferred.dataset, "pohang_canal_04")
        self.assertEqual(inferred.root, self.root)
        self.assertEqual(inferred.versions[paths_lib.OBJECT_TRACKS], "v1")

    def test_non_artifact_path_returns_none(self):
        self.assertIsNone(
            paths_lib.infer_from_artifact_path(self.root / "scratch"))

    def test_dataset_flag_disagreeing_with_run_dir_is_an_error(self):
        parser = make_parser()
        args = parser.parse_args(["--dataset", "boston_harbor_leg1"])
        with self.assertRaises(paths_lib.MissingInput):
            paths_lib.from_args(args, infer_from=self.run_dir)

    def test_recorded_run_inputs_win_over_resolution(self):
        (self.run_dir / paths_lib.RUN_META).write_text(json.dumps({
            "inputs": {"frame_landmarks":
                       str(self.root / "artifacts" / "frame_landmarks"
                           / "pohang_canal_04" / "v5"),
                       "feather": "/recorded/cat.feather"}}))
        parser = make_parser(feather=True)
        args = parser.parse_args([])
        p = paths_lib.from_args(args, infer_from=self.run_dir)
        self.assertEqual(
            p.frame_landmarks,
            self.root / "artifacts" / "frame_landmarks" / "pohang_canal_04"
            / "v5")
        self.assertEqual(p.feather, Path("/recorded/cat.feather"))

    def test_explicit_flag_wins_over_recorded_input(self):
        (self.run_dir / paths_lib.RUN_META).write_text(json.dumps({
            "inputs": {"feather": "/recorded/cat.feather"}}))
        parser = make_parser(feather=True)
        args = parser.parse_args(["--feather", "/explicit/cat.feather"])
        p = paths_lib.from_args(args, infer_from=self.run_dir)
        self.assertEqual(p.feather, Path("/explicit/cat.feather"))

    def test_run_without_record_still_fails_loudly_on_versions(self):
        # A run dir with no run_meta.json infers its own object_tracks
        # version from the path, but supplies nothing for other kinds.
        parser = make_parser()
        args = parser.parse_args([])
        p = paths_lib.from_args(args, infer_from=self.run_dir)
        with self.assertRaises(paths_lib.MissingInput):
            _ = p.frame_landmarks


class ExperimentLaneTest(unittest.TestCase):
    def test_experiment_dir(self):
        p = paths_lib.FarfieldPaths(dataset="x", root=Path("/r"))
        self.assertEqual(p.experiment_dir("260901_extent_sigma"),
                         Path("/r/runs/260901_extent_sigma"))


class RelativeToRootTest(unittest.TestCase):
    def test_inside_and_outside(self):
        root = Path("/data/farfield_matching")
        self.assertEqual(
            paths_lib.relative_to_root(root / "datasets" / "x", root),
            "datasets/x")
        self.assertEqual(
            paths_lib.relative_to_root(Path("/elsewhere/y"), root),
            "/elsewhere/y")


if __name__ == "__main__":
    unittest.main()
