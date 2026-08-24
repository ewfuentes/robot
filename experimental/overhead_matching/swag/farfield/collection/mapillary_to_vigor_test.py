"""Tests for the converter's metadata contract on synthetic, local inputs.

The load-bearing assertions cover the audit-mandated fix: this writer now
records the mount-offset frame note in `azimuth_convention`. Three docs claimed
both dataset writers carried it; only the self-collect writer did, and the 20
Mapillary datasets shipped without it carried exactly the metadata shape behind
the pohang 180-degree mount-offset incident.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield.collection import (
    mapillary_to_vigor as mtv,
)


def _synthetic_metadata(n=4, camera_type="spherical", camera_parameters=None):
    out = []
    for i in range(n):
        out.append({
            "id": str(1000 + i),
            "lat": 42.35 + 0.001 * i,
            "lng": -71.05 + 0.001 * i,
            "captured_at": 1700000000000 + 1000 * i,
            "sequence_id": "seqA" if i < n // 2 else "seqB",
            "camera_type": camera_type,
            "camera_parameters": camera_parameters,
            "computed_compass_angle": 45.0,
            "compass_angle": 47.0,
            "geometry_source": "computed",
            "width": 5760, "height": 2880,
            "pano_id": f"f{i:04d}",
        })
    return out


def _scores():
    return {
        "computed": {"field": "computed_compass_angle", "n_pairs": 3,
                     "comparison":
                         "optical_axis_world_vs_gps_course_diagnostic_only",
                     "median_abs_course_delta_deg": 4.0,
                     "mean_abs_course_delta_deg": 5.0,
                     "frac_exactly_zero": 0.0},
        "compass": {"field": "compass_angle", "n_pairs": 3,
                    "comparison":
                        "optical_axis_world_vs_gps_course_diagnostic_only",
                    "median_abs_course_delta_deg": 9.0,
                    "mean_abs_course_delta_deg": 10.0,
                    "frac_exactly_zero": 0.0},
    }


def _stats():
    return {"south": 42.35, "north": 42.353, "west": -71.05, "east": -71.047,
            "width_km": 0.25, "height_km": 0.33, "area_km2": 0.08,
            "trajectory_km": 0.4, "num_images": 4}


def _build(is_equirect=True, **overrides):
    kwargs = dict(
        dataset_name="test_ds",
        metadata=_synthetic_metadata(
            camera_type="spherical" if is_equirect else "perspective",
            camera_parameters=None if is_equirect else [0.6, 0.01, -0.02]),
        is_equirect=is_equirect,
        stats=_stats(),
        scores=_scores(),
        heading_source="computed",
        offset_info={
            "result_kind": "optical_axis_world_minus_gps_course_cw_deg",
            "mean_offset_cw_deg": 12.0,
            "circular_std_deg": 3.0,
            "n_samples": 3,
            "authority": dict(mtv.DIAGNOSTIC_AUTHORITY),
        },
        nominal_forward_info=None,
        substituted_count=0,
        image_dir_name="frames",
        num_written=4,
        resize=4096,
        min_spacing=0.0,
        jpeg_quality=95,
        max_heading_error_deg=10.0,
        max_heading_source_disagreement_deg=25.0,
        max_perspective_offset_std_deg=45.0,
    )
    kwargs.update(overrides)
    return mtv.build_pipeline_metadata(**kwargs)


class AzimuthConventionTest(unittest.TestCase):
    def test_equirect_formula_reference_is_column_0(self):
        convention = _build(is_equirect=True)["azimuth_convention"]
        self.assertFalse(convention["images_rotated"])
        self.assertIn("heading_column0_true_deg",
                      convention["world_bearing_formula"])
        self.assertIn("- 180", convention[
            "column0_from_optical_axis_formula"])

    def test_perspective_formula_is_optical_axis_explicit(self):
        convention = _build(is_equirect=False)["azimuth_convention"]
        self.assertIn("heading_optical_axis_true_deg",
                      convention["world_bearing_formula"])

    def test_never_north_aligned(self):
        for is_equirect in (True, False):
            meta = _build(is_equirect=is_equirect)
            self.assertIs(meta["north_aligned"], False)


class OffsetDiagnosticTest(unittest.TestCase):
    def test_no_mount_offset_block_is_written(self):
        # Calibration's explicit publish tool owns mount_offset (with frame +
        # applied fields); the converter writing one — in the wrong frame —
        # is how pohang shipped a 180-degree-out prior.
        meta = _build()
        self.assertNotIn("mount_offset", meta)
        self.assertNotIn("rig_offset_deg", meta)

    def test_offset_diagnostic_is_frame_tagged(self):
        diag = _build(is_equirect=True)["heading_course_diagnostic"]
        self.assertEqual(diag["result_kind"],
                         "optical_axis_world_minus_gps_course_cw_deg")
        self.assertEqual(diag["mean_offset_cw_deg"], 12.0)
        self.assertEqual(diag["authority"], mtv.DIAGNOSTIC_AUTHORITY)

    def test_wraparound_offsets_have_small_circular_spread(self):
        mean, spread = mtv._circular_mean_std_deg([179.0, -179.0])
        self.assertAlmostEqual(abs(mean), 180.0, places=6)
        self.assertLess(spread, 2.0)


class HeadingVerdictTest(unittest.TestCase):
    def test_unapproved_recommendation_never_claims_reliability(self):
        meta = _build(max_heading_error_deg=10.0)
        self.assertNotIn("heading_reliable", meta)
        self.assertEqual(
            meta["heading_source_recommendation"]["authority"],
            mtv.DIAGNOSTIC_AUTHORITY)
        self.assertIsNone(
            meta["nominal_forward_course_alignment_within_review_threshold"])

    def test_approved_nominal_forward_enables_review_threshold_only(self):
        info = {"version": "nf-v1", "bearing_camera_cw_deg": 20.0}
        meta = _build(nominal_forward_info=info,
                      max_heading_error_deg=10.0)
        self.assertIs(
            meta["nominal_forward_course_alignment_within_review_threshold"],
            True)
        self.assertNotIn("heading_reliable", meta)

    def test_perspective_has_no_gps_bearing_verdict(self):
        meta = _build(is_equirect=False)
        self.assertIsNone(
            meta["nominal_forward_course_alignment_within_review_threshold"])
        # median |computed - compass| = 2.0 <= 25 -> sources agree
        self.assertIs(meta["heading_sources_disagree"], False)
        self.assertIs(meta["camera_pans_relative_to_travel"], False)

    def test_perspective_pan_detection(self):
        meta = _build(is_equirect=False,
                      offset_info={
                          "result_kind":
                              "optical_axis_world_minus_gps_course_cw_deg",
                          "mean_offset_cw_deg": 0.0,
                          "circular_std_deg": 80.0,
                          "n_samples": 50,
                          "authority": dict(mtv.DIAGNOSTIC_AUTHORITY),
                      })
        self.assertIs(meta["camera_pans_relative_to_travel"], True)

    def test_column_zero_and_nominal_forward_formulas(self):
        self.assertEqual(mtv.optical_axis_to_column0_true_deg(0.0), 180.0)
        self.assertEqual(mtv.optical_axis_to_column0_true_deg(90.0), 270.0)
        self.assertEqual(mtv.nominal_forward_world_cw_deg(350.0, 20.0),
                         10.0)

    def test_approved_forward_avoids_the_180_degree_source_trap(self):
        metadata = _synthetic_metadata()
        # The trajectory points north-east (~39 degrees). Computed points 180
        # degrees away at its optical axis, but an approved rear-facing nominal
        # forward ray makes it the aligned source.
        for item in metadata:
            item["computed_compass_angle"] = 219.0
            item["compass_angle"] = 39.0
        computed = mtv.score_heading_source(
            metadata, "computed_compass_angle", 180.0)
        compass = mtv.score_heading_source(
            metadata, "compass_angle", 180.0)
        self.assertLess(computed["median_abs_course_delta_deg"],
                        compass["median_abs_course_delta_deg"])


class ProjectionHelpersTest(unittest.TestCase):
    def test_classify_projection_accepts_both_equirect_spellings(self):
        for camera_type in ("spherical", "equirectangular"):
            self.assertTrue(mtv.classify_projection(
                _synthetic_metadata(camera_type=camera_type)))
        self.assertFalse(mtv.classify_projection(
            _synthetic_metadata(camera_type="perspective")))

    def test_fov_from_camera_parameters(self):
        meta = {"camera_parameters": [0.5, 0, 0], "width": 4000, "height": 3000}
        hfov, vfov = mtv.fov_from_camera_parameters(meta)
        # focal 0.5 normalized by max dim: hfov = 2*atan((4000/4000)/(2*0.5)) = 90
        self.assertAlmostEqual(hfov, 90.0, places=6)
        self.assertLess(vfov, hfov)

    def test_process_single_image_takes_the_six_live_params(self):
        # The old tuple threaded rig_offset and is_equirect through to a body
        # that never read them; the audit removed both. A 6-tuple must unpack
        # cleanly, and a missing image reports an error instead of raising.
        from pathlib import Path
        meta = {"image_path": "/nonexistent/frame.jpg",
                "lat": 42.0, "lng": -71.0}
        idx, filename, error = mtv.process_single_image(
            (3, meta, Path("/tmp"), 95, None, "f0003"))
        self.assertEqual(idx, 3)
        self.assertIsNone(filename)
        self.assertIn("Failed to read", error)

    def test_failed_image_write_is_fatal_for_the_frame(self):
        class Image:
            shape = (10, 20, 3)

        meta = {"image_path": "/tmp/source.jpg",
                "lat": 42.0, "lng": -71.0}
        with mock.patch.object(mtv.cv2, "imread", return_value=Image()), \
             mock.patch.object(mtv.cv2, "imwrite", return_value=False):
            idx, filename, error = mtv.process_single_image(
                (3, meta, Path("/tmp"), 95, None, "f0003"))
        self.assertEqual(idx, 3)
        self.assertIsNone(filename)
        self.assertIn("Failed to write", error)

    def test_failed_requested_visualization_read_is_fatal(self):
        path = Path("/tmp/missing-visualization.jpg")
        with mock.patch.object(mtv.cv2, "imread", return_value=None):
            with self.assertRaisesRegex(
                    RuntimeError, "failed to read requested stored"):
                mtv.read_visualization_image(path, "stored")

    def test_conversion_staging_is_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "dataset"
            staging = mtv.create_conversion_staging(destination)
            marker = staging / "diagnostic.txt"
            marker.write_text("keep")
            with self.assertRaisesRegex(FileExistsError, "incomplete"):
                mtv.create_conversion_staging(destination)
            self.assertEqual(marker.read_text(), "keep")

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "dataset"
            destination.mkdir()
            marker = destination / "complete.txt"
            marker.write_text("keep")
            with self.assertRaisesRegex(FileExistsError, "completed"):
                mtv.create_conversion_staging(destination)
            self.assertEqual(marker.read_text(), "keep")


class SequencePositionContractTest(unittest.TestCase):
    @staticmethod
    def _sidecar(root: Path, stem: str, position):
        (root / f"{stem}.jpg").write_bytes(b"fixture")
        record = {"captured_at": 1000}
        if position is not None:
            record["sequence_position"] = position
        (root / f"{stem}.json").write_text(json.dumps(record))

    @staticmethod
    def _load(root: Path, stems: list[str]):
        manifest = {"expected": [
            {"id": stem, "sequence_position": index, "stem": stem}
            for index, stem in enumerate(stems)
        ]}
        with mock.patch.object(
                mtv.extract_stitch, "validate_download_directory",
                return_value=manifest):
            return mtv.load_sequence_metadata(root)

    def test_every_sidecar_requires_sequence_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._sidecar(root, "a", 0)
            self._sidecar(root, "b", None)
            with self.assertRaisesRegex(ValueError, "every frame sidecar"):
                self._load(root, ["a", "b"])

    def test_duplicate_sequence_positions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._sidecar(root, "a", 2)
            self._sidecar(root, "b", 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self._load(root, ["a", "b"])

    def test_sidecars_are_sorted_by_authoritative_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._sidecar(root, "a", 1)
            self._sidecar(root, "b", 0)
            loaded = self._load(root, ["b", "a"])
            self.assertEqual([item["sequence_position"] for item in loaded],
                             [0, 1])


class ConversionReuseContractTest(unittest.TestCase):
    def test_manifest_binds_recipe_input_and_all_output_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset.incomplete"
            root.mkdir()
            (root / "payload.txt").write_text("version one")
            input_download = {
                "path": "/source",
                "manifest_sha256": "a" * 64,
                "content_digest": "b" * 64,
                "source_manifest": {"path": "/request", "sha256": "c" * 64},
            }
            inputs = {"sequence_dir": "/source"}
            config = {"dataset_name": "example", "visualize": False}
            mtv.provenance.write(
                root, generator=mtv.CONVERSION_GENERATOR, inputs=inputs,
                config=config,
                content_digest=mtv.artifact.sha256_directory(root),
                extra={"input_download": input_download, "complete": True},
            )

            mtv.validate_completed_conversion(
                root, input_download=input_download, inputs=inputs,
                config=config, allow_incomplete=True)
            (root / "payload.txt").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "content digest"):
                mtv.validate_completed_conversion(
                    root, input_download=input_download, inputs=inputs,
                    config=config, allow_incomplete=True)


if __name__ == "__main__":
    unittest.main()
