import argparse
import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.data import farfield_paths as fp


def make_dataset(root: Path, name: str, *, video: str | None = None,
                 extra: dict | None = None) -> Path:
    """Lay down the minimum a dataset needs to be resolvable."""
    base = root / "datasets" / name
    (base / "panorama").mkdir(parents=True)
    (base / "landmarks").mkdir(parents=True)
    (base / "landmarks" / "v1_trimmed.feather").write_bytes(b"")
    meta = {"dataset_name": name, "num_images": 3}
    if video is not None:
        meta["video"] = {"source_video": video}
    if extra:
        meta.update(extra)
    (base / "pipeline_metadata.json").write_text(json.dumps(meta))
    return base


class FarfieldPathsTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_resolves_dataset_and_artifact_lanes(self):
        make_dataset(self.root, "leg2")
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root)

        self.assertEqual(paths.dataset_base, self.root / "datasets" / "leg2")
        self.assertEqual(paths.panorama_dir,
                         self.root / "datasets" / "leg2" / "panorama")
        self.assertEqual(
            paths.frame_landmarks,
            self.root / "artifacts" / "frame_landmarks" / "leg2" / "v1")
        self.assertEqual(
            paths.pinhole_images,
            self.root / "artifacts" / "pinhole_images" / "leg2" / "v1")
        # Named through DEFAULT_CATALOG rather than spelled out, so bumping the
        # default catalog is a one-line change instead of a test failure -- but
        # the shape of the path still gets asserted.
        self.assertEqual(
            paths.feather,
            self.root / "datasets" / "leg2" / "landmarks"
            / f"{fp.DEFAULT_CATALOG}.feather")
        self.assertEqual(paths.sam2_checkpoint,
                         self.root / "models" / "sam2" / "sam2.1_hiera_large.pt")

    def test_artifact_version_is_per_kind(self):
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root,
                                 versions={fp.FRAME_LANDMARKS: "v2"})
        self.assertTrue(str(paths.frame_landmarks).endswith("leg2/v2"))
        # An unrelated kind keeps the default rather than following along.
        self.assertTrue(str(paths.pinhole_images).endswith("leg2/v1"))

    def test_unknown_artifact_kind_is_rejected(self):
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root)
        with self.assertRaises(ValueError):
            paths.artifact("frame_landmark")  # missing plural

    def test_tracks_stage_dirs_live_under_the_artifact(self):
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root)
        self.assertEqual(paths.tracks_stage("m3_tracks"),
                         paths.object_tracks / "m3_tracks")
        self.assertEqual(paths.tracks_runs_root,
                         paths.object_tracks / "m3_tracks" / "runs")

    def test_video_comes_from_dataset_metadata(self):
        make_dataset(self.root, "leg2",
                     video="raw_material/collect/videos/hull_to_hingham.mp4")
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root)
        self.assertEqual(
            paths.video,
            self.root / "raw_material" / "collect" / "videos" /
            "hull_to_hingham.mp4")

    def test_video_strips_trailing_retention_note(self):
        # The boston metadata carries a human note after the path; it must not
        # end up as part of the filename.
        make_dataset(
            self.root, "leg2",
            video=("raw_material/collect/videos/hull_to_hingham.mp4 "
                   "(not retained; ~38 GB originals)"))
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root)
        self.assertEqual(paths.video.name, "hull_to_hingham.mp4")

    def test_absolute_video_path_is_left_alone(self):
        make_dataset(self.root, "leg2", video="/elsewhere/x.mp4")
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root)
        self.assertEqual(paths.video, Path("/elsewhere/x.mp4"))

    def test_missing_video_metadata_explains_itself(self):
        make_dataset(self.root, "leg2")  # no video key at all
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root)
        with self.assertRaises(fp.MissingInput) as ctx:
            _ = paths.video
        self.assertIn("--video", str(ctx.exception))

    def test_missing_dataset_metadata_names_the_dataset(self):
        paths = fp.FarfieldPaths(dataset="nope", root=self.root)
        with self.assertRaises(fp.MissingInput) as ctx:
            paths.metadata()
        self.assertIn("nope", str(ctx.exception))

    def test_overrides_win_over_resolution(self):
        make_dataset(self.root, "leg2", video="raw_material/c/v/a.mp4")
        elsewhere = self.root / "scratch" / "lm"
        paths = fp.FarfieldPaths(
            dataset="leg2", root=self.root,
            overrides={"frame_landmarks": elsewhere,
                       "video": Path("/tmp/other.mp4")})
        self.assertEqual(paths.frame_landmarks, elsewhere)
        self.assertEqual(paths.video, Path("/tmp/other.mp4"))
        # Un-overridden paths still resolve normally.
        self.assertTrue(str(paths.pinhole_images).endswith("leg2/v1"))

    def test_require_reports_every_missing_input_at_once(self):
        make_dataset(self.root, "leg2")
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root)
        with self.assertRaises(fp.MissingInput) as ctx:
            paths.require("panorama_dir", "frame_landmarks", "sam2_checkpoint")
        message = str(ctx.exception)
        # panorama_dir exists; the other two do not, and both are named.
        self.assertNotIn("panorama_dir", message)
        self.assertIn("frame_landmarks", message)
        self.assertIn("sam2_checkpoint", message)

    def test_require_surfaces_unresolvable_video(self):
        make_dataset(self.root, "leg2")  # no video key
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root)
        with self.assertRaises(fp.MissingInput) as ctx:
            paths.require("video")
        self.assertIn("video", str(ctx.exception))

    def test_write_manifest_records_provenance(self):
        paths = fp.FarfieldPaths(dataset="leg2", root=self.root)
        written = paths.write_manifest(
            fp.PINHOLE_IMAGES,
            generator="//x:panorama_to_pinhole",
            config={"res_x": 2048},
            inputs=["datasets/leg2"],
            notes="four faces per pano")
        self.assertEqual(written.parent, paths.pinhole_images)
        manifest = json.loads(written.read_text())
        self.assertEqual(manifest["kind"], "pinhole_images")
        self.assertEqual(manifest["dataset"], "leg2")
        self.assertEqual(manifest["version"], "v1")
        self.assertEqual(manifest["config"], {"res_x": 2048})
        self.assertIn("git_commit", manifest)
        self.assertIn("created", manifest)

    def test_relative_to_root_shortens_inside_paths_only(self):
        inside = self.root / "datasets" / "leg2"
        self.assertEqual(fp.relative_to_root(inside, self.root),
                         "datasets/leg2")
        self.assertEqual(fp.relative_to_root(Path("/elsewhere/x"), self.root),
                         "/elsewhere/x")


class InferenceTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_dir(self, dataset="leg3", version="v1", run="r001"):
        path = (self.root / "artifacts" / "object_tracks" / dataset / version /
                "m3_tracks" / "runs" / run)
        path.mkdir(parents=True)
        return path

    def test_infers_dataset_and_version_from_run_dir(self):
        paths = fp.infer_from_artifact_path(self.run_dir("leg3", "v2"))
        self.assertIsNotNone(paths)
        self.assertEqual(paths.dataset, "leg3")
        self.assertEqual(paths.root, self.root.resolve())
        # The version the run actually lives in, not the default.
        self.assertEqual(paths.version(fp.OBJECT_TRACKS), "v2")

    def test_infers_from_the_artifact_dir_itself(self):
        path = self.root / "artifacts" / "frame_landmarks" / "leg2" / "v1"
        path.mkdir(parents=True)
        paths = fp.infer_from_artifact_path(path)
        self.assertEqual(paths.dataset, "leg2")
        self.assertEqual(paths.version(fp.FRAME_LANDMARKS), "v1")

    def test_returns_none_outside_an_artifact_lane(self):
        self.assertIsNone(fp.infer_from_artifact_path(self.root / "scratch"))
        # A directory literally named artifacts but with an unknown kind.
        odd = self.root / "artifacts" / "not_a_kind" / "leg3" / "v1"
        odd.mkdir(parents=True)
        self.assertIsNone(fp.infer_from_artifact_path(odd))

    def test_from_args_infers_when_dataset_omitted(self):
        run = self.run_dir("leg3")
        parser = argparse.ArgumentParser()
        fp.add_arguments(parser)
        paths = fp.from_args(parser.parse_args([]), infer_from=run)
        self.assertEqual(paths.dataset, "leg3")
        self.assertEqual(paths.dataset_base,
                         self.root.resolve() / "datasets" / "leg3")

    def test_from_args_rejects_a_dataset_that_contradicts_the_run_dir(self):
        # The exact mistake this replaces: leg3's run dir with leg1's frames.
        run = self.run_dir("leg3")
        parser = argparse.ArgumentParser()
        fp.add_arguments(parser)
        args = parser.parse_args(["--dataset", "leg1"])
        with self.assertRaises(fp.MissingInput) as ctx:
            fp.from_args(args, infer_from=run)
        message = str(ctx.exception)
        self.assertIn("leg1", message)
        self.assertIn("leg3", message)

    def test_agreeing_dataset_and_run_dir_is_fine(self):
        run = self.run_dir("leg3")
        parser = argparse.ArgumentParser()
        fp.add_arguments(parser)
        args = parser.parse_args(["--dataset", "leg3"])
        self.assertEqual(fp.from_args(args, infer_from=run).dataset, "leg3")

    def test_explicit_root_beats_the_inferred_one(self):
        run = self.run_dir("leg3")
        parser = argparse.ArgumentParser()
        fp.add_arguments(parser)
        args = parser.parse_args(["--farfield_root", "/elsewhere"])
        self.assertEqual(fp.from_args(args, infer_from=run).root,
                         Path("/elsewhere"))


class RecordedRunInputsTest(unittest.TestCase):
    """A run's own record of what it was built from outranks re-resolution."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.run = (self.root / "artifacts" / "object_tracks" / "leg2" / "v1"
                    / "m3_tracks" / "runs" / "r002")
        self.run.mkdir(parents=True)

    def write_meta(self, inputs):
        (self.run / "run_meta.json").write_text(json.dumps({"inputs": inputs}))

    def parse(self, argv):
        parser = argparse.ArgumentParser()
        fp.add_arguments(parser)
        return fp.from_args(parser.parse_args(argv), infer_from=self.run)

    def test_v2_frame_landmarks_survive_into_later_stages(self):
        # The failure this prevents: a run built on v2 whose audit/merge/match
        # silently read v1 back, because v1 is the lane default.
        v2 = self.root / "artifacts" / "frame_landmarks" / "leg2" / "v2"
        self.write_meta({"frame_landmarks": str(v2)})
        self.assertEqual(self.parse([]).frame_landmarks, v2)

    def test_without_a_record_normal_resolution_applies(self):
        paths = self.parse([])
        self.assertTrue(str(paths.frame_landmarks).endswith("leg2/v1"))

    def test_explicit_flag_beats_the_record(self):
        self.write_meta({"frame_landmarks": "/recorded/v2"})
        paths = self.parse(["--landmark_base", "/explicit/v9"])
        self.assertEqual(paths.frame_landmarks, Path("/explicit/v9"))

    def test_video_and_dataset_base_are_recovered_too(self):
        self.write_meta({"video": "/raw/legs/hull_to_hingham.mp4",
                         "dataset_base": "/data/datasets/leg2"})
        paths = self.parse([])
        self.assertEqual(paths.video, Path("/raw/legs/hull_to_hingham.mp4"))
        self.assertEqual(paths.dataset_base, Path("/data/datasets/leg2"))

    def test_unparseable_or_absent_meta_is_ignored(self):
        self.assertEqual(fp.recorded_run_inputs(self.run), {})
        (self.run / "run_meta.json").write_text("{not json")
        self.assertEqual(fp.recorded_run_inputs(self.run), {})

    def test_unknown_recorded_keys_are_not_smuggled_in(self):
        self.write_meta({"frame_landmarks": "/x/v2", "something_else": "/y"})
        self.assertEqual(set(fp.recorded_run_inputs(self.run)),
                         {"frame_landmarks"})


class ArgParsingTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def parse(self, argv, **kwargs):
        parser = argparse.ArgumentParser()
        fp.add_arguments(parser, **kwargs)
        return fp.from_args(parser.parse_args(argv))

    def test_dataset_name_is_enough(self):
        paths = self.parse(["--dataset", "leg3",
                            "--farfield_root", str(self.root)])
        self.assertEqual(paths.dataset, "leg3")
        self.assertEqual(paths.dataset_base, self.root / "datasets" / "leg3")

    def test_dataset_base_alone_infers_the_name(self):
        # Out-of-tree dataset dir: artifact lanes still need a key, and the
        # directory name is it.
        paths = self.parse(["--dataset_base", "/scratch/leg9",
                            "--farfield_root", str(self.root)])
        self.assertEqual(paths.dataset, "leg9")
        self.assertEqual(paths.dataset_base, Path("/scratch/leg9"))
        self.assertTrue(str(paths.frame_landmarks).endswith("leg9/v1"))

    def test_neither_dataset_nor_base_is_an_error(self):
        with self.assertRaises(fp.MissingInput):
            self.parse(["--farfield_root", str(self.root)])

    def test_version_flags_are_threaded_through(self):
        paths = self.parse(["--dataset", "leg3",
                            "--farfield_root", str(self.root),
                            "--frame_landmarks_version", "v2",
                            "--pinhole_version", "v3"], pinhole=True)
        self.assertTrue(str(paths.frame_landmarks).endswith("leg3/v2"))
        self.assertTrue(str(paths.pinhole_images).endswith("leg3/v3"))

    def test_optional_flags_absent_unless_requested(self):
        parser = argparse.ArgumentParser()
        fp.add_arguments(parser)
        args = parser.parse_args(["--dataset", "leg3"])
        self.assertFalse(hasattr(args, "video"))
        self.assertFalse(hasattr(args, "checkpoint"))
        # ...and from_args tolerates their absence.
        self.assertEqual(fp.from_args(args).dataset, "leg3")

    def test_catalog_flag_selects_the_feather(self):
        paths = self.parse(["--dataset", "leg3",
                            "--farfield_root", str(self.root),
                            "--catalog", "v1"], feather=True)
        self.assertEqual(paths.feather.name, "v1.feather")


if __name__ == "__main__":
    unittest.main()


class DefaultCatalogTest(unittest.TestCase):
    """The default catalog is the problem definition for every matching run, so
    changing it changes what past numbers mean. Pinned here so the change is
    deliberate."""

    def test_default_is_v2_trimmed(self):
        self.assertEqual(fp.DEFAULT_CATALOG, "v2_trimmed")

    def test_explicit_catalog_overrides_the_default(self):
        paths = fp.FarfieldPaths(dataset="leg2", root=Path("/r"),
                                             catalog="v1_trimmed")
        self.assertEqual(paths.feather.name, "v1_trimmed.feather")
