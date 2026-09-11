"""Horizon-plane fit, levelling, and temporal gate on synthetic panoramas."""

import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    level_equirect_video as lev,
)

W, H = 960, 480


def tilt_rotation(beta_deg, lean_az_deg):
    """Frame whose up leans by beta toward azimuth lean_az (cw from forward)."""
    n = lev.unit_vectors(lean_az_deg, 90.0 - beta_deg)
    return lev.level_rotation(n), n          # R @ n = z


def render_world(R_level, shoreline=True, cloud=True):
    """Camera frame image of a world with sky above a horizon plane at el -1 deg
    (true frame), green ground below, a dark 'shoreline' band 5-8 deg below the
    horizon over half the azimuths, and a grey cloud underside above."""
    d_cam = lev.pano_dirs(W, H).reshape(-1, 3)
    d_true = d_cam @ R_level.T                     # R_level @ d_cam, row form
    _, el = lev.az_el_from_vectors(d_true)
    az, _ = lev.az_el_from_vectors(d_true)
    img = np.zeros((H * W, 3), np.uint8)
    sky = el > -1.0
    img[sky] = (235, 205, 150)                      # BGR: pale bright sky
    img[~sky] = (60, 140, 70)                       # green ground
    if shoreline:
        band = (~sky) & (el < -5.0) & (el > -8.0) & (az < 180.0)
        img[band] = (90, 70, 40)                    # dark water/shore band
    if cloud:
        blob = sky & (el > 12.0) & (el < 20.0) & (az > 200.0) & (az < 300.0)
        img[blob] = (150, 150, 150)                 # grey cloud underside
    rng = np.random.default_rng(1)
    noise = rng.integers(-6, 7, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img.reshape(H, W, 3)


class ConventionParityTest(unittest.TestCase):

    def test_vectorised_maths_matches_geometry(self):
        rng = np.random.default_rng(0)
        xs = rng.uniform(0, W, 200)
        ys = rng.uniform(0, H, 200)
        d = lev.pano_dirs(W, H)
        for x, y in zip(xs, ys):
            az_ref, el_ref = geo.direction_from_pano_px(x, y, W, H)
            az, el = lev.az_el_from_vectors(lev.unit_vectors(az_ref, el_ref))
            self.assertAlmostEqual(az, az_ref, places=9)
            self.assertAlmostEqual(el, el_ref, places=9)
            # pixel-centre grid agrees with the scalar owner
            az_c, el_c = geo.direction_from_pano_px(int(x) + 0.5, int(y) + 0.5, W, H)
            az_v, el_v = lev.az_el_from_vectors(d[int(y), int(x)])
            self.assertAlmostEqual(az_v, az_c, places=9)
            self.assertAlmostEqual(el_v, el_c, places=9)
            px, py = geo.pano_px_from_direction(az_ref, el_ref, W, H)
            vx, vy = lev.dirs_to_px(lev.unit_vectors(az_ref, el_ref), W, H)
            self.assertAlmostEqual(vx + 0.5, px, places=6)
            self.assertAlmostEqual(vy + 0.5, py, places=6)


class HorizonFitTest(unittest.TestCase):

    def test_recovers_tilt_despite_shoreline_and_cloud(self):
        rng = np.random.default_rng(0)
        for beta, lean in ((12.0, 80.0), (5.0, 300.0), (17.0, 10.0)):
            R_level, n_true = tilt_rotation(beta, lean)
            img = render_world(R_level)
            fit = lev.fit_horizon(img, rng)
            self.assertIsNotNone(fit, f"no fit at beta={beta}")
            self.assertLess(lev.angle_between_deg(fit.n, n_true), 0.3, f"beta={beta}")
            self.assertAlmostEqual(lev.tilt_params(fit.n)["tilt_deg"], beta, delta=0.3)
            self.assertAlmostEqual(fit.horizon_el_deg, -1.0, delta=0.6)
            self.assertGreater(fit.coverage, 0.5)
            levelled = lev.level_image_cpu(img, lev.level_rotation(fit.n))
            refit = lev.fit_horizon(levelled, rng)
            self.assertIsNotNone(refit)
            self.assertLess(lev.tilt_params(refit.n)["tilt_deg"], 0.15)

    def test_level_rotation_has_no_yaw(self):
        _, n = tilt_rotation(15.0, 60.0)
        R = lev.level_rotation(n)
        np.testing.assert_allclose(R @ n, [0, 0, 1], atol=1e-9)
        # axis of the rotation is horizontal
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        self.assertAlmostEqual(axis[2], 0.0, places=9)

    def test_no_fit_on_featureless_frame(self):
        img = np.full((H, W, 3), (60, 140, 70), np.uint8)
        self.assertIsNone(lev.fit_horizon(img, np.random.default_rng(0)))


class TemporalGateTest(unittest.TestCase):

    def test_outliers_and_gaps_are_interpolated(self):
        ns = []
        for i in range(20):
            beta = 10.0 + 0.05 * i
            ns.append(lev.unit_vectors(90.0, 90.0 - beta))
        ns[7] = lev.unit_vectors(90.0, 90.0 - 16.0)          # 6 degree jump
        ns[12] = None                                        # failed fit
        post = [0.1] * 20
        post[15] = 0.9                                       # failed self-check
        applied, status = lev.temporal_gate(ns, post)
        self.assertEqual(status[7], "continuity_gate")
        self.assertEqual(status[12], "no_fit")
        self.assertEqual(status[15], "self_check_gate")
        self.assertEqual(status.count("fit"), 17)
        for i in (7, 12, 15):
            expect = lev.unit_vectors(90.0, 90.0 - (10.0 + 0.05 * i))
            self.assertLess(lev.angle_between_deg(applied[i], expect), 0.1)

    def test_all_gated_raises(self):
        with self.assertRaises(ValueError):
            lev.temporal_gate([None, None], [None, None])

    def test_slerp_endpoints(self):
        a = lev.unit_vectors(0.0, 80.0)
        b = lev.unit_vectors(90.0, 80.0)
        np.testing.assert_allclose(lev.slerp(a, b, 0.0), a, atol=1e-12)
        np.testing.assert_allclose(lev.slerp(a, b, 1.0), b, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(lev.slerp(a, b, 0.5)), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
