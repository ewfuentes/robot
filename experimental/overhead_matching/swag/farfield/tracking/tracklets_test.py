import math
import unittest

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.tracking import tracklets

PANO_W = 7680
PARAMS = tracklets.TrackletParams(epoch_keyframes=5, bearing_sigma_deg=1.0)


def make_track(track_id, birth, boxes_by_kf, close_reason="end_of_range"):
    """boxes_by_kf: {keyframe: (x0, y0, x1, y1)} in window coords with a
    zero window origin, so pano coords == window coords."""
    return {
        "track_id": track_id,
        "birth_keyframe": birth,
        "end_keyframe": max(boxes_by_kf) if boxes_by_kf else birth,
        "close_reason": close_reason,
        "records": [
            {"keyframe": kf, "mask_bbox_window": list(box),
             "window_origin": [0, 0], "action": "propagate"}
            for kf, box in sorted(boxes_by_kf.items())
        ],
    }


def centred_box(pano_x, width=80, y0=1000, y1=1400):
    return (pano_x - width / 2, y0, pano_x + width / 2, y1)


class BearingSeriesTest(unittest.TestCase):
    def test_series_uses_the_camera_frame_owner(self):
        # Mask centred on the pano centre column -> azimuth 0; on the
        # three-quarter column -> azimuth 90.
        track = make_track(1, 0, {0: centred_box(PANO_W / 2),
                                  1: centred_box(0.75 * PANO_W)})
        series = tracklets.bearing_series(track, PANO_W)
        self.assertEqual([kf for kf, _, _ in series], [0, 1])
        self.assertAlmostEqual(series[0][1], 0.0)
        self.assertAlmostEqual(series[1][1], 90.0)
        self.assertAlmostEqual(series[0][2], 80 / PANO_W * 360.0)

    def test_valid_segments_crop_the_drifted_tail(self):
        # Audit segments are relative to birth; a track born at kf 10 with a
        # valid segment [0, 2] keeps keyframes 10-12 and drops 13+.
        boxes = {kf: centred_box(PANO_W / 2) for kf in range(10, 16)}
        track = make_track(2, 10, boxes)
        series = tracklets.bearing_series(
            track, PANO_W, valid_segments=[{"start_t": 0, "end_t": 2}])
        self.assertEqual([kf for kf, _, _ in series], [10, 11, 12])

    def test_records_without_masks_are_skipped(self):
        track = make_track(3, 0, {0: centred_box(100)})
        track["records"].append({"keyframe": 1, "mask_bbox_window": None,
                                 "window_origin": [0, 0], "action": "gap"})
        self.assertEqual(len(tracklets.bearing_series(track, PANO_W)), 1)

    def test_window_origin_offsets_apply(self):
        track = {
            "track_id": 4, "birth_keyframe": 0, "end_keyframe": 0,
            "close_reason": "end_of_range",
            "records": [{"keyframe": 0, "mask_bbox_window": [0, 0, 100, 100],
                         "window_origin": [PANO_W / 2 - 50, 500],
                         "action": "propagate"}],
        }
        series = tracklets.bearing_series(track, PANO_W)
        self.assertAlmostEqual(series[0][1], 0.0)  # centred after offset


class FuseBearingsTest(unittest.TestCase):
    def test_epoch_bucketing(self):
        series = [(kf, 10.0, 4.0) for kf in range(10)]
        fused = tracklets.fuse_bearings(series, PARAMS)
        # 10 keyframes at epoch 5 -> 2 fused measurements, anchored at the
        # middle keyframe of each bucket.
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0][0], 2)
        self.assertEqual(fused[1][0], 7)

    def test_circular_mean_across_wrap(self):
        series = [(0, 359.0, 4.0), (1, 1.0, 4.0)]
        fused = tracklets.fuse_bearings(series, PARAMS)
        self.assertAlmostEqual(fused[0][1], 0.0, places=6)

    def test_kappa_reflects_width_not_count(self):
        # A wide object is a soft bearing regardless of how many keyframes
        # were fused; kappa must not grow with the bucket size.
        narrow = tracklets.fuse_bearings([(0, 10.0, 1.0)], PARAMS)[0][2]
        wide = tracklets.fuse_bearings([(0, 10.0, 20.0)], PARAMS)[0][2]
        self.assertGreater(narrow, wide)
        one = tracklets.fuse_bearings([(0, 10.0, 4.0)], PARAMS)[0][2]
        many = tracklets.fuse_bearings(
            [(kf, 10.0, 4.0) for kf in range(5)], PARAMS)[0][2]
        self.assertAlmostEqual(one, many)

    def test_kappa_matches_documented_formula(self):
        fused = tracklets.fuse_bearings([(0, 10.0, 8.0)], PARAMS)
        sigma = math.hypot(1.0, 8.0 / 4.0)
        self.assertAlmostEqual(fused[0][2], 1.0 / math.radians(sigma) ** 2)

    def test_empty_series(self):
        self.assertEqual(tracklets.fuse_bearings([], PARAMS), [])


class BuildMeasurementsTest(unittest.TestCase):
    def setUp(self):
        self.tracks = {
            1: make_track(1, 0, {kf: centred_box(PANO_W / 2)
                                 for kf in range(6)}),
            2: make_track(2, 0, {kf: centred_box(0.75 * PANO_W)
                                 for kf in range(3)}),
            3: make_track(3, 0, {0: centred_box(1000)}),
        }

    def test_audit_membership_is_the_gate(self):
        # Track 3 was never audited: no canonical semantics, no measurement.
        audits = {1: {"valid_segments": None}, 2: {"valid_segments": None}}
        measurements = tracklets.build_measurements(
            self.tracks, audits, PANO_W, PARAMS)
        self.assertEqual({m.tracklet_id for m in measurements}, {"T1", "T2"})

    def test_measurements_sorted_by_anchor_then_id(self):
        audits = {1: {}, 2: {}}
        measurements = tracklets.build_measurements(
            self.tracks, audits, PANO_W, PARAMS)
        keys = [(m.anchor_keyframe_idx, m.tracklet_id) for m in measurements]
        self.assertEqual(keys, sorted(keys))

    def test_audited_track_missing_from_tracks_is_skipped(self):
        audits = {1: {}, 99: {}}
        measurements = tracklets.build_measurements(
            self.tracks, audits, PANO_W, PARAMS)
        self.assertEqual({m.tracklet_id for m in measurements}, {"T1"})

    def test_segments_flow_through_to_the_series(self):
        audits = {1: {"valid_segments": [{"start_t": 0, "end_t": 1}]}}
        measurements = tracklets.build_measurements(
            self.tracks, audits, PANO_W, PARAMS)
        # Keyframes 0-1 only -> a single epoch bucket.
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0].anchor_keyframe_idx, 1)

    def test_bearing_is_camera_frame(self):
        audits = {2: {}}
        m = tracklets.build_measurements(self.tracks, audits, PANO_W,
                                         PARAMS)[0]
        self.assertAlmostEqual(
            m.bearing_camera_deg,
            geo.azimuth_of_pano_column(0.75 * PANO_W, PANO_W))


class ParityWithMergedPipelineTest(unittest.TestCase):
    """The old m6 measurements, minus the weld: for a run where no tracks
    were merged (the overwhelmingly common case), the fused bearings are
    identical to what m6's fuse_bearings produced."""

    def test_single_track_parity(self):
        # Reference values computed with the old track_merge.fuse_bearings
        # semantics: epoch 5, sigma 1.0, circular mean per bucket, kappa
        # from hypot(sigma, width/4).
        boxes = {kf: centred_box(PANO_W / 2 + 40 * kf) for kf in range(7)}
        track = make_track(7, 0, boxes)
        series = tracklets.bearing_series(track, PANO_W)
        fused = tracklets.fuse_bearings(series, PARAMS)

        def old_fuse(series, epoch_keyframes, bearing_sigma_deg):
            fused, bucket = [], []
            start_kf = series[0][0]
            def flush(bucket):
                if not bucket:
                    return
                sin_sum = sum(math.sin(math.radians(a)) for _, a, _ in bucket)
                cos_sum = sum(math.cos(math.radians(a)) for _, a, _ in bucket)
                mean_az = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
                mean_width = sum(w for _, _, w in bucket) / len(bucket)
                anchor = bucket[len(bucket) // 2][0]
                sigma = math.hypot(bearing_sigma_deg, mean_width / 4.0)
                fused.append((anchor, mean_az, 1.0 / math.radians(sigma) ** 2))
            for entry in series:
                if entry[0] - start_kf >= epoch_keyframes:
                    flush(bucket)
                    bucket = []
                    start_kf = entry[0]
                bucket.append(entry)
            flush(bucket)
            return fused

        reference = old_fuse(series, 5, 1.0)
        self.assertEqual(len(fused), len(reference))
        for (a1, az1, k1), (a2, az2, k2) in zip(fused, reference):
            self.assertEqual(a1, a2)
            self.assertAlmostEqual(az1, az2, places=9)
            self.assertAlmostEqual(k1, k2, places=9)


if __name__ == "__main__":
    unittest.main()
