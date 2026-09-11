import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.dataset_tools import anonymize_video as anonymize


class AnonymizeVideoTest(unittest.TestCase):

    def test_integer_frame_selection_retains_original_frame_grid(self):
        expression, step = anonymize.frame_selection(30.0, 3.0)
        self.assertEqual(expression, "select=not(mod(n\\,10))")
        self.assertEqual(step, 10)
        self.assertEqual(anonymize.expected_output_frames({
            "media_fps": 30.0,
            "nb_frames": 32584,
            "duration_s": 1086.133333,
        }, 3.0), 3259)

    def test_raw_reader_does_not_duplicate_selected_frames_back_to_30_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "source.mkv"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc=size=64x32:rate=30:duration=1", "-c:v", "ffv1",
                str(video),
            ], check=True)
            info = anonymize.probe_video(video)
            frames = list(anonymize.RawVideoReader(
                video, info, output_fps=3.0, width=64, height=32))
            self.assertEqual(len(frames), 3)

    def test_clip_is_end_exclusive_on_output_grid(self):
        video = {
            "media_fps": 30.0,
            "nb_frames": 30 * 3000,
            "duration_s": 3000.0,
        }
        clip = anonymize.clip_metadata(video, 3.0, 130.0, 2505.0)
        self.assertEqual(clip, {
            "start_s": 130.0,
            "end_s": 2505.0,
            "start_frame": 390,
            "end_frame_exclusive": 7515,
            "frame_count": 7125,
        })

    def test_raw_reader_applies_clip_after_frame_rate_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "source.mkv"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc=size=64x32:rate=30:duration=2", "-c:v", "ffv1",
                str(video),
            ], check=True)
            info = anonymize.probe_video(video)
            frames = list(anonymize.RawVideoReader(
                video, info, output_fps=3.0, width=64, height=32,
                start_frame=1, end_frame=5))
            self.assertEqual(len(frames), 4)

    def test_cfr_fast_seek_matches_baseline_at_subsecond_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "source.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc=size=64x32:rate=3:duration=8", "-c:v", "libx264",
                "-g", "24", "-keyint_min", "24", "-sc_threshold", "0",
                "-pix_fmt", "yuv420p", str(video),
            ], check=True)
            info = anonymize.probe_video(video)
            baseline = list(anonymize.RawVideoReader(
                video, info, output_fps=3.0, width=64, height=32,
                start_frame=13, end_frame=19))
            fast = list(anonymize.RawVideoReader(
                video, info, output_fps=3.0, width=64, height=32,
                start_frame=13, end_frame=19, cfr_fast_seek=True))
            self.assertEqual(len(baseline), 6)
            self.assertTrue(all(
                np.array_equal(left, right)
                for left, right in zip(baseline, fast, strict=True)))

    def test_cfr_fast_seek_matches_baseline_with_integer_rate_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "source.mkv"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc=size=64x32:rate=30:duration=8", "-c:v", "ffv1",
                str(video),
            ], check=True)
            info = anonymize.probe_video(video)
            baseline = list(anonymize.RawVideoReader(
                video, info, output_fps=3.0, width=64, height=32,
                start_frame=13, end_frame=19))
            fast = list(anonymize.RawVideoReader(
                video, info, output_fps=3.0, width=64, height=32,
                start_frame=13, end_frame=19, cfr_fast_seek=True))
            self.assertEqual(len(baseline), 6)
            self.assertTrue(all(
                np.array_equal(left, right)
                for left, right in zip(baseline, fast, strict=True)))

    def test_cfr_fast_seek_rejects_noninteger_rate_selection(self):
        with self.assertRaisesRegex(ValueError, "exact multiple"):
            anonymize.cfr_fast_seek_fps({"media_fps": 30.0}, 4.0)

    def test_video_writer_stages_until_explicit_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review.mp4"
            writer = anonymize.RawVideoWriter(
                output, 64, 32, 3.0, review=True)
            writer.write(np.zeros((32, 64, 3), dtype=np.uint8))
            writer.close(publish=False)
            self.assertFalse(output.exists())
            self.assertTrue(writer.incomplete.is_file())
            writer.publish()
            self.assertTrue(output.is_file())
            self.assertFalse(writer.incomplete.exists())

    def test_nvenc_profile_uses_dedicated_nvidia_video_encoders(self):
        profile = anonymize.video_encoder_profile("nvenc")
        self.assertEqual(profile["backend"], "nvenc")
        self.assertEqual(profile["full"]["codec"], "hevc_nvenc")
        self.assertEqual(profile["review"]["codec"], "h264_nvenc")
        self.assertEqual(profile["full"]["ffmpeg_args"], [
            "-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq",
            "-rc", "vbr", "-cq", "18", "-b:v", "0",
        ])
        self.assertEqual(profile["review"]["ffmpeg_args"], [
            "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
            "-rc", "vbr", "-cq", "23", "-b:v", "0",
        ])
        with self.assertRaisesRegex(ValueError, "unsupported video encoder"):
            anonymize.video_encoder_profile("mystery")

    def test_software_encoder_profile_preserves_legacy_settings(self):
        profile = anonymize.video_encoder_profile("software")
        self.assertEqual(profile["full"]["ffmpeg_args"], [
            "-c:v", "libx265", "-preset", "ultrafast", "-crf", "18",
            "-x265-params", "pools=16:frame-threads=2:log-level=error",
        ])
        self.assertEqual(profile["review"]["ffmpeg_args"], [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        ])

    def test_circular_box_splits_across_equirectangular_seam(self):
        boxes = anonymize.circular_box(
            [0, 10, 30, 30], start_x=90, full_width=100, full_height=50)
        self.assertEqual(boxes, [
            [0.9, 0.2, 1.0, 0.6],
            [0.0, 0.2, 0.2, 0.6],
        ])

    def test_temporal_densification_and_manual_region(self):
        raw = [
            {"frame_index": 0, "video_t_s": 0.0, "detections": []},
            {"frame_index": 1, "video_t_s": 1 / 3, "detections": [{
                "category": "face", "source": "test", "confidence": 0.8,
                "box": [0.1, 0.1, 0.2, 0.2],
            }]},
            {"frame_index": 2, "video_t_s": 2 / 3, "detections": []},
        ]
        manual = [{
            "id": "fixed", "category": "face",
            "box": [0.7, 0.2, 0.9, 0.5],
            "start_s": 0.0, "end_s": 0.5, "reason": "test",
        }]
        output = anonymize.densify(raw, 1, manual, 3.0)
        self.assertEqual(len(output), 3)
        self.assertEqual([len(row["detections"]) for row in output], [2, 2, 1])
        self.assertEqual(output[0]["detections"][0]["category"], "face")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            anonymize.densify(raw, -1, manual, 3.0)

    def test_policy_revision_reuses_bound_raw_detections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan_dir = root / "scan"
            output_dir = root / "policy"
            scan_dir.mkdir()
            raw_path = scan_dir / "detections.raw.jsonl"
            raw = [
                {"frame_index": 0, "video_t_s": 0.0, "detections": [{
                    "category": "face", "source": "test", "confidence": 0.9,
                    "box": [0.1, 0.1, 0.2, 0.2],
                }]},
                {"frame_index": 1, "video_t_s": 1 / 3, "detections": []},
            ]
            anonymize.write_jsonl(raw_path, raw)
            (scan_dir / "anonymization_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "status": "scanned",
                "output_fps": 3.0,
                "frame_count": len(raw),
                "files": {"raw_ledger": {
                    "path": raw_path.name,
                    "sha256": anonymize.sha256_file(raw_path),
                }},
            }))
            anonymize.apply_policy(argparse.Namespace(
                scan_dir=scan_dir,
                output_dir=output_dir,
                manual_regions=None,
                temporal_radius_frames=1,
            ))
            applied = anonymize.read_jsonl(output_dir / "detections.jsonl")
            self.assertEqual(len(applied[1]["detections"]), 1)
            revised = json.loads(
                (output_dir / "anonymization_manifest.json").read_text())
            self.assertEqual(revised["status"], "scanned")
            self.assertIn("policy_parent", revised)
            self.assertEqual(
                revised["policy_parent"]["manifest_sha256"],
                anonymize.sha256_file(
                    scan_dir / "anonymization_manifest.json"))
            self.assertEqual(
                (output_dir / "detections.raw.jsonl").read_bytes(),
                raw_path.read_bytes())
            self.assertEqual(
                revised["files"]["raw_ledger"]["sha256"],
                anonymize.sha256_file(output_dir / "detections.raw.jsonl"))
            self.assertEqual(
                revised["files"]["applied_ledger"]["sha256"],
                anonymize.sha256_file(output_dir / "detections.jsonl"))

    def test_policy_revision_atomically_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan_dir = root / "scan"
            output_dir = root / "policy"
            scan_dir.mkdir()
            output_dir.mkdir()
            sentinel = output_dir / "owned_by_other_process"
            sentinel.write_text("preserve me")
            raw_path = scan_dir / "detections.raw.jsonl"
            raw = [{"frame_index": 0, "video_t_s": 0.0,
                    "detections": []}]
            anonymize.write_jsonl(raw_path, raw)
            (scan_dir / "anonymization_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "status": "scanned",
                "output_fps": 3.0,
                "frame_count": len(raw),
                "files": {"raw_ledger": {
                    "path": raw_path.name,
                    "sha256": anonymize.sha256_file(raw_path),
                }},
            }))
            with self.assertRaises(FileExistsError):
                anonymize.apply_policy(argparse.Namespace(
                    scan_dir=scan_dir,
                    output_dir=output_dir,
                    manual_regions=None,
                    temporal_radius_frames=1,
                ))
            self.assertEqual(sentinel.read_text(), "preserve me")

    def test_manual_regions_use_original_source_time_after_clipping(self):
        raw = [{
            "frame_index": 0,
            "video_t_s": 0.0,
            "source_video_t_s": 130.0,
            "detections": [],
        }]
        manual = [{
            "id": "source_clock", "category": "face",
            "box": [0.1, 0.1, 0.2, 0.2],
            "start_s": 129.5, "end_s": 130.5, "reason": "test",
        }]
        output = anonymize.densify(raw, 0, manual, 3.0)
        self.assertEqual(len(output[0]["detections"]), 1)

    def test_strong_blur_changes_only_requested_region(self):
        generator = np.random.default_rng(7)
        image = generator.integers(0, 256, (100, 200, 3), dtype=np.uint8)
        original = image.copy()
        anonymize.strong_blur(image, [{
            "category": "face", "box": [0.25, 0.20, 0.75, 0.80],
        }])
        np.testing.assert_array_equal(image[:20], original[:20])
        np.testing.assert_array_equal(image[80:], original[80:])
        self.assertFalse(np.array_equal(image[20:80, 50:150],
                                        original[20:80, 50:150]))

    def test_manual_regions_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.json"
            path.write_text('[{"box":[0.9,0.1,0.2,0.4]}]')
            with self.assertRaises(ValueError):
                anonymize.read_manual_regions(path)
            path.write_text('[{"box":[0.1,0.1,0.2,NaN]}]')
            with self.assertRaises(ValueError):
                anonymize.read_manual_regions(path)

    def test_review_decision_is_additive(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest = output_dir / "anonymization_manifest.json"
            blurred = output_dir / "blurred.mp4"
            review = output_dir / "review.mp4"
            review_html = output_dir / "review.html"
            ledger = output_dir / "detections.jsonl"
            for path in (blurred, review, review_html, ledger):
                path.write_bytes(path.name.encode())
            manifest.write_text(json.dumps({
                "status": "rendered_pending_review",
                "render": {
                    "output_video": str(blurred),
                    "output_video_sha256": anonymize.sha256_file(blurred),
                    "review_video_sha256": anonymize.sha256_file(review),
                    "review_html_sha256": anonymize.sha256_file(review_html),
                },
                "review": {"video": review.name, "html": review_html.name},
                "files": {"applied_ledger": {
                    "path": ledger.name,
                    "sha256": anonymize.sha256_file(ledger),
                }},
            }))
            before = anonymize.sha256_file(manifest)
            anonymize.mark_review(argparse.Namespace(
                output_dir=output_dir,
                decision="approved",
                reviewer="reviewer",
                note="watched end to end",
            ))
            self.assertEqual(anonymize.sha256_file(manifest), before)
            decision = json.loads(
                (output_dir / "review_decision.json").read_text())
            self.assertEqual(decision["anonymization_manifest_sha256"], before)
            self.assertEqual(decision["status"], "approved")

    def test_review_html_includes_native_resolution_ledger_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.html"
            anonymize._write_review_html(
                path, "review.mp4", "../blurred.mp4", "detections.jsonl",
                5.0, 130.0, 3.0, 7680, 3840)
            html = path.read_text()
            self.assertIn("Native-resolution inspector (required)", html)
            self.assertIn('full.src=fullUrl', html)
            self.assertIn('fetch(ledgerUrl)', html)
            self.assertIn('const fullWidth=7680, fullHeight=3840', html)
            self.assertNotIn('__REVIEW_URL__', html)

    def test_no_clobber_file_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            output = root / "output"
            staging.write_text("new")
            output.write_text("existing")
            with self.assertRaises(FileExistsError):
                anonymize.publish_file_no_clobber(staging, output)
            self.assertEqual(staging.read_text(), "new")
            self.assertEqual(output.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
