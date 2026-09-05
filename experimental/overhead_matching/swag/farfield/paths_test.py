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
                             "--catalogs_version", "v3_trimmed"])
        self.assertEqual(
            p2.feather,
            self.root / "artifacts" / "catalogs" / "boston_harbor_leg2"
            / "v3_trimmed" / "catalog.feather")

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

    def test_metadata_video_cannot_escape_the_farfield_root(self):
        meta = self.root / "datasets" / "boston_harbor_leg2" / \
            "pipeline_metadata.json"
        for source in ("../outside.mp4", "/outside.mp4"):
            with self.subTest(source=source):
                meta.write_text(json.dumps({"video": {
                    "source_video": source}}))
                p = self.paths_for(["--dataset", "boston_harbor_leg2"])
                with self.assertRaises(paths_lib.MissingInput):
                    _ = p.video

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
        self.artifact_dir = (self.root / "artifacts" / "object_tracks"
                             / "pohang_canal_04" / "v1")
        self.artifact_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_infers_dataset_root_and_version(self):
        inferred = paths_lib.infer_from_artifact_path(self.artifact_dir)
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
            paths_lib.from_args(args, infer_from=self.artifact_dir)

    def test_inference_supplies_only_the_containing_artifact_version(self):
        parser = make_parser()
        args = parser.parse_args([])
        p = paths_lib.from_args(args, infer_from=self.artifact_dir)
        self.assertEqual(p.object_tracks, self.artifact_dir)
        with self.assertRaises(paths_lib.MissingInput):
            _ = p.frame_landmarks


class ExperimentLaneTest(unittest.TestCase):
    def test_build_and_localization_run_lanes_are_distinct(self):
        p = paths_lib.FarfieldPaths(dataset="x", root=Path("/r"))
        self.assertEqual(p.build_dir("b001"), Path("/r/builds/x/b001"))
        self.assertEqual(p.experiment_dir("260901_extent_sigma"),
                         Path("/r/runs/260901_extent_sigma"))

    def test_all_artifact_kinds_have_uniform_version_directories(self):
        versions = {kind: "v9" for kind in paths_lib.ARTIFACT_KINDS}
        p = paths_lib.FarfieldPaths(
            dataset="x", root=Path("/r"), versions=versions)
        for kind in paths_lib.ARTIFACT_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(
                    p.artifact(kind), Path("/r/artifacts") / kind / "x" / "v9")

    def test_lane_components_are_path_free_identifiers(self):
        for dataset in ("../escape", "/absolute"):
            with self.subTest(dataset=dataset):
                with self.assertRaises(paths_lib.PathContractError):
                    paths_lib.FarfieldPaths(dataset=dataset, root=Path("/r"))

        p = paths_lib.FarfieldPaths(dataset="x", root=Path("/r"))
        for build_name in ("../escape", "/absolute"):
            with self.subTest(build_name=build_name):
                with self.assertRaises(paths_lib.PathContractError):
                    p.build_dir(build_name)
        with self.assertRaises(paths_lib.PathContractError):
            p.experiment_dir("../../escape")
        with self.assertRaises(paths_lib.PathContractError):
            p.artifact(paths_lib.OBJECT_TRACKS, "../escape")


class RelativeToRootTest(unittest.TestCase):
    def test_inside_and_outside(self):
        root = Path("/data/farfield_matching")
        self.assertEqual(
            paths_lib.relative_to_root(root / "datasets" / "x", root),
            "datasets/x")
        self.assertEqual(
            paths_lib.relative_to_root(Path("/elsewhere/y"), root),
            "/elsewhere/y")


class DatasetSourceDigestTest(unittest.TestCase):
    def test_snapshot_changes_when_any_consumed_dataset_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_base = Path(temporary)
            (dataset_base / "panorama").mkdir()
            metadata = dataset_base / "pipeline_metadata.json"
            frames_gps = dataset_base / "frames_gps.csv"
            panorama = dataset_base / "panorama" / "f0000,1,2,.jpg"
            metadata.write_text("{}")
            frames_gps.write_text("idx\n0\n")
            panorama.write_bytes(b"jpeg")
            baseline = paths_lib.dataset_source_digests(dataset_base)
            self.assertEqual(set(baseline), set(
                paths_lib.DATASET_SOURCE_DIGEST_KEYS))
            for path, replacement in (
                    (metadata, b'{"changed":true}'),
                    (frames_gps, b"idx\n0\n1\n"),
                    (panorama, b"different jpeg")):
                with self.subTest(path=path.name):
                    original = path.read_bytes()
                    path.write_bytes(replacement)
                    self.assertNotEqual(
                        paths_lib.dataset_source_digests(dataset_base),
                        baseline)
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
