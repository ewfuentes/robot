"""Transactional publication tests for dataset review views."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    make_dataset_timelapse as timelapse,
)


class TimelapsePublicationTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dataset = Path(self.temporary.name) / "example"
        panorama = self.dataset / "panorama"
        panorama.mkdir(parents=True)
        rows = ["frame_file,latitude,longitude,video_t_s,dist_m"]
        for index in range(2):
            filename = f"f{index:04d}.jpg"
            Image.new("RGB", (16, 8), (index * 20, 40, 60)).save(
                panorama / filename)
            rows.append(
                f"{filename},{42.0 + index * 0.001},-71.0,{index},"
                f"{index * 10}")
        (self.dataset / "frames_gps.csv").write_text("\n".join(rows) + "\n")
        (self.dataset / "pipeline_metadata.json").write_text("{}\n")

    @staticmethod
    def plot(dataset, lats, lons, times, dists, out):
        del dataset, lats, lons, times, dists
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (20, 40, 60)).save(out, format="PNG")

    @staticmethod
    def video(paths, lats, lons, out, width, fps, max_frames):
        del paths, lats, lons, width, fps, max_frames
        out.write_bytes(b"\x00\x00\x00\x18ftypmp42video")

    def render(self):
        with mock.patch.object(
                timelapse, "stage_plot", side_effect=self.plot) as plot, \
             mock.patch.object(
                 timelapse, "stage_video", side_effect=self.video) as video:
            reference = timelapse.render(
                self.dataset, width=640, fps=12,
                max_frames=100, skip_video=False)
        return reference, plot, video

    def test_pair_is_published_as_one_complete_typed_directory(self):
        reference, plot, video = self.render()
        output = timelapse.view_output_dir(self.dataset)

        self.assertEqual(Path(reference.path), output.resolve())
        artifact.open_artifact(
            output,
            expected_kind=timelapse.REVIEW_KIND,
            expected_dataset="example",
            expected_version=timelapse.REVIEW_VERSION)
        manifest = artifact.load_manifest(output)
        self.assertEqual(manifest.declared_outputs, (
            timelapse.TIMELAPSE_NAME, timelapse.TRAJECTORY_NAME))
        self.assertEqual(set(manifest.config["input_digests"]), {
            "pipeline_metadata", "frames_gps", "panorama_directory"})
        self.assertFalse(output.with_name("timelapse.incomplete").exists())
        plot.assert_called_once()
        video.assert_called_once()

    def test_exact_completed_identity_is_reused_without_rendering(self):
        first, _, _ = self.render()
        with mock.patch.object(timelapse, "stage_plot") as plot, \
             mock.patch.object(timelapse, "stage_video") as video:
            second = timelapse.render(
                self.dataset, width=640, fps=12,
                max_frames=100, skip_video=False)
        self.assertEqual(first, second)
        plot.assert_not_called()
        video.assert_not_called()

    def test_failed_second_output_leaves_no_visible_review_artifact(self):
        def fail_video(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("encoder failed")

        with mock.patch.object(
                timelapse, "stage_plot", side_effect=self.plot), \
             mock.patch.object(
                 timelapse, "stage_video", side_effect=fail_video):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                timelapse.render(
                    self.dataset, width=640, fps=12,
                    max_frames=100, skip_video=False)

        output = timelapse.view_output_dir(self.dataset)
        self.assertFalse(output.exists())
        staging = output.with_name("timelapse.incomplete")
        self.assertTrue(staging.is_dir())
        self.assertTrue((staging / timelapse.TRAJECTORY_NAME).is_file())
        self.assertFalse((staging / timelapse.TIMELAPSE_NAME).exists())

    def test_changed_source_cannot_reuse_or_replace_completed_pair(self):
        self.render()
        with (self.dataset / "frames_gps.csv").open("a") as stream:
            stream.write("\n")
        with mock.patch.object(timelapse, "stage_plot") as plot, \
             mock.patch.object(timelapse, "stage_video") as video:
            with self.assertRaisesRegex(ValueError, "different identity"):
                timelapse.render(
                    self.dataset, width=640, fps=12,
                    max_frames=100, skip_video=False)
        plot.assert_not_called()
        video.assert_not_called()

    def test_legacy_loose_view_blocks_new_publication(self):
        legacy = self.dataset / "_manifests" / timelapse.TRAJECTORY_NAME
        legacy.parent.mkdir()
        legacy.write_bytes(b"old")
        with self.assertRaisesRegex(FileExistsError, "legacy timelapse"):
            timelapse.render(
                self.dataset, width=640, fps=12,
                max_frames=100, skip_video=False)
        self.assertFalse(timelapse.view_output_dir(self.dataset).exists())


if __name__ == "__main__":
    unittest.main()
