"""Tests for the converter's metadata contract, on synthetic inputs (no I/O,
no network).

The load-bearing assertions cover the audit-mandated fix: this writer now
records the mount-offset frame note in `azimuth_convention`. Three docs claimed
both dataset writers carried it; only the self-collect writer did, and the 20
Mapillary datasets shipped without it carried exactly the metadata shape behind
the pohang 180-degree mount-offset incident.
"""

import unittest

from experimental.overhead_matching.swag.farfield import geometry
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
            "heading_used": 45.0,
            "pano_id": f"f{i:04d}",
        })
    return out


def _scores():
    return {
        "computed": {"field": "computed_compass_angle", "n_pairs": 3,
                     "median_err_deg": 4.0, "mean_err_deg": 5.0,
                     "frac_exactly_zero": 0.0},
        "compass": {"field": "compass_angle", "n_pairs": 3,
                    "median_err_deg": 9.0, "mean_err_deg": 10.0,
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
        offset_info={"offset_deg": 12.0, "std_deg": 3.0, "n_samples": 3},
        substituted_count=0,
        image_dir_name="frames",
        num_written=4,
        resize=4096,
        min_spacing=0.0,
        jpeg_quality=95,
        max_heading_error_deg=10.0,
        max_heading_source_disagreement_deg=25.0,
        max_perspective_offset_std_deg=45.0,
        skip_heading_validation=False,
    )
    kwargs.update(overrides)
    return mtv.build_pipeline_metadata(**kwargs)


class AzimuthConventionTest(unittest.TestCase):
    def test_equirect_carries_the_mount_offset_frame_note(self):
        convention = _build(is_equirect=True)["azimuth_convention"]
        # Verbatim, so a reader of the dataset alone gets the full contract —
        # convention strings are exported constants, never restated.
        self.assertEqual(convention["mount_offset_frame"],
                         geometry.MOUNT_OFFSET_CONVENTION)
        self.assertEqual(convention["frame_if_derived_from_formula"],
                         "column_0_NOT_usable_as_mount_offset")

    def test_equirect_formula_reference_is_column_0(self):
        convention = _build(is_equirect=True)["azimuth_convention"]
        self.assertEqual(convention["heading_deg_is_bearing_of"], "column_0")
        self.assertFalse(convention["images_rotated"])
        self.assertIn("formula", convention)

    def test_perspective_carries_the_note_with_its_own_frame_tag(self):
        convention = _build(is_equirect=False)["azimuth_convention"]
        self.assertEqual(convention["mount_offset_frame"],
                         geometry.MOUNT_OFFSET_CONVENTION)
        self.assertEqual(convention["frame_if_derived_from_formula"],
                         "optical_axis_NOT_usable_as_mount_offset")
        self.assertEqual(convention["heading_deg_is_bearing_of"],
                         "optical_axis")

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
        diag = _build(is_equirect=True)["heading_vs_travel_offset_diagnostic"]
        self.assertEqual(diag["frame"], "column_0_NOT_usable_as_mount_offset")
        self.assertEqual(diag["offset_deg"], 12.0)
        self.assertIn("NOT a mount_offset", diag["note"])
        diag_p = _build(is_equirect=False)["heading_vs_travel_offset_diagnostic"]
        self.assertEqual(diag_p["frame"],
                         "optical_axis_NOT_usable_as_mount_offset")


class HeadingVerdictTest(unittest.TestCase):
    def test_equirect_heading_reliable_gates_on_threshold(self):
        self.assertIs(_build(max_heading_error_deg=10.0)["heading_reliable"],
                      True)   # median 4.0 <= 10
        self.assertIs(_build(max_heading_error_deg=2.0)["heading_reliable"],
                      False)  # median 4.0 > 2

    def test_perspective_has_no_gps_bearing_verdict(self):
        meta = _build(is_equirect=False)
        self.assertIsNone(meta["heading_reliable"])
        # median |computed - compass| = 2.0 <= 25 -> sources agree
        self.assertIs(meta["heading_sources_disagree"], False)
        self.assertIs(meta["camera_pans_relative_to_travel"], False)

    def test_perspective_pan_detection(self):
        meta = _build(is_equirect=False,
                      offset_info={"offset_deg": 0.0, "std_deg": 80.0,
                                   "n_samples": 50})
        self.assertIs(meta["camera_pans_relative_to_travel"], True)


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


if __name__ == "__main__":
    unittest.main()
