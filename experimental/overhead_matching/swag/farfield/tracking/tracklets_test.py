import math
import unittest

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.tracking import tracklets

PANO_W = 7680
PARAMS = tracklets.TrackletParams(
    epoch_keyframes=5, bearing_sigma_deg=1.0)


def make_track(track_id, birth, boxes_by_kf, close_reason="end_of_range"):
    end = max(boxes_by_kf) if boxes_by_kf else birth
    return {
        "track_id": track_id,
        "birth_keyframe": birth,
        "end_keyframe": end,
        "last_keyframe": end,
        "close_reason": close_reason,
        "records": [
            {"keyframe": keyframe, "mask_bbox_window": list(box),
             "window_origin": [0, 0], "action": "propagate"}
            for keyframe, box in sorted(boxes_by_kf.items())
        ],
    }


def centred_box(pano_x, width=80, y0=1000, y1=1400):
    return (pano_x - width / 2, y0, pano_x + width / 2, y1)


def audit_for(track, verdict="keep", segments=None):
    if segments is None:
        segments = [{
            "start_t": 0,
            "end_t": track["end_keyframe"] - track["birth_keyframe"],
        }]
    return {
        "verdict": verdict,
        "single_object": verdict == "keep",
        "drop_reason": ("dynamic_object" if verdict == "drop" else "none"),
        "valid_segments": segments,
        "confidence": "high",
    }


class BoundAudits(dict):
    def __init__(self, values, tracks):
        super().__init__(values)
        self.provenance_by_track = {
            track_id: {
                "source_tracks_artifact_id": "object_tracks/demo/v2",
                "source_tracks_sha256": "a" * 64,
                "source_track_sha256": tracklets._canonical_sha256(
                    tracks[track_id]),
                "audit_key": f"T{track_id}",
            }
            for track_id in values
        }


class BearingSeriesTest(unittest.TestCase):
    def test_series_uses_camera_cw_frame_and_box_midpoint(self):
        track = make_track(
            1, 0, {
                0: centred_box(PANO_W / 2),
                1: centred_box(0.75 * PANO_W),
            })
        series = tracklets.bearing_series(track, PANO_W)
        self.assertEqual([keyframe for keyframe, _, _ in series], [0, 1])
        self.assertAlmostEqual(series[0][1], 0.0)
        self.assertAlmostEqual(series[1][1], 90.0)
        self.assertAlmostEqual(series[0][2], 80 / PANO_W * 360.0)

    def test_valid_segments_crop_drifted_tail(self):
        boxes = {
            keyframe: centred_box(PANO_W / 2)
            for keyframe in range(10, 16)
        }
        track = make_track(2, 10, boxes)
        series = tracklets.bearing_series(
            track, PANO_W,
            valid_segments=[{"start_t": 0, "end_t": 2}])
        self.assertEqual(
            [keyframe for keyframe, _, _ in series], [10, 11, 12])

    def test_empty_valid_segments_mean_no_observations(self):
        track = make_track(3, 0, {0: centred_box(100)})
        self.assertEqual(
            tracklets.bearing_series(track, PANO_W, valid_segments=[]), [])

    def test_records_without_masks_are_skipped(self):
        track = make_track(4, 0, {0: centred_box(100)})
        track["end_keyframe"] = 1
        track["records"].append({
            "keyframe": 1, "mask_bbox_window": None,
            "window_origin": [0, 0], "action": "gap",
        })
        self.assertEqual(len(tracklets.bearing_series(track, PANO_W)), 1)

    def test_window_origin_offsets_apply(self):
        track = {
            "track_id": 5,
            "birth_keyframe": 0,
            "end_keyframe": 0,
            "records": [{
                "keyframe": 0,
                "mask_bbox_window": [0, 0, 100, 100],
                "window_origin": [PANO_W / 2 - 50, 500],
            }],
        }
        series = tracklets.bearing_series(track, PANO_W)
        self.assertAlmostEqual(series[0][1], 0.0)


class AcceptedTrackletTest(unittest.TestCase):
    def setUp(self):
        self.track1 = make_track(
            1, 10, {
                keyframe: centred_box(PANO_W / 2)
                for keyframe in range(10, 16)
            })
        self.track2 = make_track(
            2, 0, {keyframe: centred_box(1000) for keyframe in range(3)})
        self.track3 = make_track(3, 0, {0: centred_box(2000)})

    def test_one_policy_accepts_keep_and_partial_but_excludes_drop(self):
        tracks = {1: self.track1, 2: self.track2, 3: self.track3}
        audits = BoundAudits({
            1: audit_for(self.track1),
            2: audit_for(
                self.track2, "keep_partial",
                [{"start_t": 1, "end_t": 2}]),
            3: audit_for(self.track3, "drop", []),
        }, tracks)
        accepted = tracklets.build_accepted_tracklets(
            tracks, audits)
        self.assertEqual([item.local_id for item in accepted], ["T1", "T2"])
        self.assertEqual(
            accepted[0].tracklet_id,
            f"object_tracks/demo/v2@sha256:{'a' * 64}#T1")
        self.assertIs(accepted[0].source_track, self.track1)
        self.assertIs(accepted[0].audit, audits[1])
        self.assertEqual(
            accepted[1].valid_segments[0].start_keyframe_idx, 1)
        self.assertEqual(
            accepted[0].provenance["source_track_sha256"],
            tracklets._canonical_sha256(self.track1))

    def test_orphaned_audit_is_an_error(self):
        with self.assertRaisesRegex(
                tracklets.TrackletContractError, "no source track"):
            tracklets.build_accepted_tracklets(
                {1: self.track1}, {99: audit_for(self.track1)})

    def test_empty_accepted_segments_never_restore_the_full_track(self):
        for verdict in ("keep", "keep_partial"):
            with self.subTest(verdict=verdict):
                with self.assertRaisesRegex(
                        tracklets.TrackletContractError, "no valid segment"):
                    tracklets.build_accepted_tracklets(
                        {2: self.track2},
                        {2: audit_for(self.track2, verdict, [])})

    def test_unsupported_tail_is_valid_but_not_part_of_audit_lifetime(self):
        track = make_track(4, 4, {
            4: centred_box(1000),
            5: centred_box(1010),
            6: centred_box(1020),
        })
        track["last_keyframe"] = 8
        track["records"].extend([
            {
                "keyframe": 7,
                "mask_bbox_window": list(centred_box(1030)),
                "window_origin": [0, 0],
                "action": "unsupported",
            },
            {
                "keyframe": 8,
                "mask_bbox_window": None,
                "window_origin": [0, 0],
                "action": "mask_dead",
            },
        ])
        accepted = tracklets.build_accepted_tracklets(
            {4: track}, {4: audit_for(
                track, segments=[{"start_t": 0, "end_t": 4}])})
        self.assertEqual(
            [(segment.start_keyframe_idx, segment.end_keyframe_idx)
             for segment in accepted[0].valid_segments],
            [(4, 6)])
        self.assertEqual(accepted[0].quality["valid_segment_clips"], [{
            "index": 0,
            "submitted_end_t": 4,
            "accepted_end_t": 2,
            "reason": "unsupported_lifecycle_tail",
        }])
        observations = tracklets.build_camera_bearing_observations(
            accepted, PANO_W, bearing_sigma_deg=1.25)
        self.assertEqual(
            [observation.keyframe_idx for observation in observations],
            [4, 5, 6])

    def test_segment_entirely_in_unsupported_tail_is_rejected(self):
        track = make_track(4, 4, {
            4: centred_box(1000),
            5: centred_box(1010),
        })
        track["last_keyframe"] = 7
        track["records"].extend([
            {
                "keyframe": keyframe,
                "mask_bbox_window": list(centred_box(1020)),
                "window_origin": [0, 0],
                "action": "unsupported",
            }
            for keyframe in (6, 7)
        ])
        with self.assertRaisesRegex(
                tracklets.TrackletContractError,
                "contains no supported evidence"):
            tracklets.build_accepted_tracklets(
                {4: track}, {4: audit_for(
                    track, segments=[{"start_t": 2, "end_t": 3}])})

    def test_record_after_last_propagated_keyframe_is_rejected(self):
        track = make_track(4, 4, {4: centred_box(1000)})
        track["records"].append({
            "keyframe": 5,
            "mask_bbox_window": list(centred_box(1010)),
            "window_origin": [0, 0],
            "action": "unsupported",
        })
        with self.assertRaisesRegex(
                tracklets.TrackletContractError, "outside its lifecycle"):
            tracklets.build_accepted_tracklets(
                {4: track}, {4: audit_for(track)})

    def test_last_propagated_keyframe_cannot_precede_supported_end(self):
        track = make_track(4, 4, {
            4: centred_box(1000),
            5: centred_box(1010),
        })
        track["last_keyframe"] = 4
        with self.assertRaisesRegex(
                tracklets.TrackletContractError,
                "precedes its supported end_keyframe"):
            tracklets.build_accepted_tracklets(
                {4: track}, {4: audit_for(track)})

    def test_keep_may_trim_unreliable_same_object_spans(self):
        accepted = tracklets.build_accepted_tracklets(
            {1: self.track1},
            {1: audit_for(
                self.track1, "keep",
                [{"start_t": 0, "end_t": 1},
                 {"start_t": 4, "end_t": 5}])})
        self.assertEqual(
            [(segment.start_keyframe_idx, segment.end_keyframe_idx)
             for segment in accepted[0].valid_segments],
            [(10, 11), (14, 15)])
        observations = tracklets.build_camera_bearing_observations(
            accepted, PANO_W, bearing_sigma_deg=1.25)
        self.assertEqual(
            [observation.keyframe_idx for observation in observations],
            [10, 11, 14, 15])

    def test_contradictory_verdict_fields_are_rejected(self):
        audit = audit_for(self.track2, "keep_partial")
        audit["single_object"] = True
        with self.assertRaisesRegex(
                tracklets.TrackletContractError,
                "keep_partial requires"):
            tracklets.build_accepted_tracklets(
                {2: self.track2}, {2: audit})

    def test_bound_source_track_digest_is_checked_again_at_join(self):
        audits = BoundAudits(
            {1: audit_for(self.track1)}, {1: self.track1})
        changed = dict(self.track1)
        changed["close_reason"] = "changed_after_audit"
        with self.assertRaisesRegex(
                tracklets.TrackletContractError, "source-track digest"):
            tracklets.build_accepted_tracklets({1: changed}, audits)

    def test_segment_bounds_order_and_non_overlap_are_validated(self):
        invalid = [
            [{"start_t": -1, "end_t": 1}],
            [{"start_t": 2, "end_t": 1}],
            [{"start_t": 0, "end_t": 6}],
            [{"start_t": 0, "end_t": 2},
             {"start_t": 2, "end_t": 4}],
            [{"start_t": 3, "end_t": 4},
             {"start_t": 0, "end_t": 1}],
        ]
        for verdict in ("keep", "keep_partial"):
            for segments in invalid:
                with self.subTest(verdict=verdict, segments=segments):
                    with self.assertRaises(tracklets.TrackletContractError):
                        tracklets.build_accepted_tracklets(
                            {1: self.track1},
                            {1: audit_for(
                                self.track1, verdict, segments)})


class CameraBearingObservationTest(unittest.TestCase):
    def test_every_valid_keyframe_is_preserved_and_segments_split_groups(self):
        track = make_track(
            7, 10, {
                keyframe: centred_box(PANO_W / 2 + 10 * keyframe)
                for keyframe in range(10, 16)
            })
        audits = BoundAudits({
            7: audit_for(
                track, "keep_partial",
                [{"start_t": 0, "end_t": 1},
                 {"start_t": 4, "end_t": 5}]),
        }, {7: track})
        accepted = tracklets.build_accepted_tracklets({7: track}, audits)
        observations = tracklets.build_camera_bearing_observations(
            accepted, PANO_W, bearing_sigma_deg=1.25)
        self.assertEqual(
            [observation.keyframe_idx for observation in observations],
            [10, 11, 14, 15])
        self.assertEqual(
            len({observation.correlation_group
                 for observation in observations}), 2)
        self.assertEqual(
            observations[0].correlation_group,
            observations[1].correlation_group)
        self.assertNotEqual(
            observations[1].correlation_group,
            observations[2].correlation_group)
        self.assertTrue(
            all(observation.sigma_deg == 1.25
                for observation in observations))
        self.assertTrue(
            all(observation.tracklet_id == accepted[0].tracklet_id
                for observation in observations))
        self.assertAlmostEqual(
            observations[0].angular_width_deg, 80 / PANO_W * 360.0)


def observation(keyframe, azimuth, width=4.0, group="segment-0",
                tracklet_id="global#T1", sigma=1.0, range_max_m=None):
    return tracklets.CameraBearingObservation(
        tracklet_id=tracklet_id,
        keyframe_idx=keyframe,
        bearing_camera_cw_deg=azimuth,
        angular_width_deg=width,
        sigma_deg=sigma,
        correlation_group=group,
        range_max_m=range_max_m)


class _Detection:
    def __init__(self, bucket):
        self.additional_tags = ([] if bucket is None
                                else [["distance_estimate", bucket]])


class RangeCapTest(unittest.TestCase):
    def _track(self):
        return {
            "track_id": 3, "birth_keyframe": 10, "birth_obs_id": "b",
            "records": [
                {"keyframe": 10, "action": "birth", "supports": []},
                {"keyframe": 11, "action": "continue_mask", "supports": [
                    {"class": "merge_superset", "obs_id": "s1"},
                    {"class": "none", "obs_id": "ignored"}]},
                {"keyframe": 12, "action": "continue_mask", "supports": [
                    {"class": "weak", "obs_id": "s2a"},
                    {"class": "weak", "obs_id": "s2b"}]},
                {"keyframe": 13, "action": "continue_mask", "supports": [
                    {"class": "split_child", "obs_id": "s3"}]},
                {"keyframe": 14, "action": "unsupported", "supports": []},
            ]}

    def test_caps_follow_evidence_supports_and_take_the_tightest(self):
        obs = {"b": _Detection("100m_to_500m"),
               "s1": _Detection("under_100m"),
               "ignored": _Detection("under_100m"),
               "s2a": _Detection("2km_to_10km"),
               "s2b": _Detection("500m_to_2km"),
               "s3": _Detection(None)}
        caps = tracklets.range_caps_by_keyframe(self._track(), obs)
        # 13 has a detection without a bucket; 14 has no detection at all.
        self.assertEqual(caps, {10: 500.0, 11: 100.0, 12: 2000.0})

    def test_over_10km_is_no_cap_and_unknown_bucket_is_an_error(self):
        obs = {"b": _Detection("over_10km"), "s1": _Detection("over_10km"),
               "s2a": _Detection(None), "s2b": _Detection(None),
               "s3": _Detection(None)}
        self.assertEqual(
            tracklets.range_caps_by_keyframe(self._track(), obs), {})
        obs["s1"] = _Detection("about_a_mile")
        with self.assertRaisesRegex(tracklets.TrackletContractError,
                                    "unknown distance_estimate"):
            tracklets.range_caps_by_keyframe(self._track(), obs)

    def test_missing_observation_is_an_error(self):
        with self.assertRaisesRegex(tracklets.TrackletContractError,
                                    "unknown observation"):
            tracklets.range_caps_by_keyframe(self._track(), {})

    def test_observation_cap_must_be_null_or_positive(self):
        observation(0, 1.0, range_max_m=None)
        observation(0, 1.0, range_max_m=100.0)
        for bad in (0.0, -5.0, float("nan"), True):
            with self.assertRaises(tracklets.TrackletContractError):
                observation(0, 1.0, range_max_m=bad)

    def test_epoch_fusion_keeps_the_tightest_cap(self):
        fused = tracklets.epoch_fused_compat_v1(
            [observation(0, 10.0, range_max_m=2000.0),
             observation(1, 10.0, range_max_m=None),
             observation(2, 10.0, range_max_m=100.0),
             observation(5, 10.0), observation(6, 10.0)], PARAMS)
        self.assertEqual([m.range_max_m for m in fused], [100.0, None])


class EpochFusedCompatV1Test(unittest.TestCase):
    def test_epoch_bucketing_uses_middle_real_keyframe(self):
        observations = [observation(keyframe, 10.0)
                        for keyframe in range(10)]
        fused = tracklets.epoch_fused_compat_v1(observations, PARAMS)
        self.assertEqual(
            [measurement.anchor_keyframe_idx for measurement in fused],
            [2, 7])

    def test_correlation_groups_are_never_fused_together(self):
        observations = [
            observation(0, 10.0, group="segment-0"),
            observation(1, 10.0, group="segment-0"),
            observation(2, 20.0, group="segment-1"),
            observation(3, 20.0, group="segment-1"),
        ]
        fused = tracklets.epoch_fused_compat_v1(observations, PARAMS)
        self.assertEqual(len(fused), 2)
        self.assertEqual(
            [measurement.anchor_keyframe_idx for measurement in fused],
            [1, 3])

    def test_circular_mean_across_wrap(self):
        fused = tracklets.epoch_fused_compat_v1(
            [observation(0, 359.0), observation(1, 1.0)], PARAMS)
        self.assertAlmostEqual(
            fused[0].bearing_camera_cw_deg, 0.0, places=6)

    def test_kappa_matches_v1_formula_and_not_observation_count(self):
        one = tracklets.epoch_fused_compat_v1(
            [observation(0, 10.0, width=8.0)], PARAMS)[0]
        many = tracklets.epoch_fused_compat_v1(
            [observation(keyframe, 10.0, width=8.0)
             for keyframe in range(5)], PARAMS)[0]
        sigma = math.hypot(1.0, 8.0 / 4.0)
        self.assertAlmostEqual(one.kappa, 1.0 / math.radians(sigma) ** 2)
        self.assertAlmostEqual(one.kappa, many.kappa)

    def test_duplicate_keyframe_inside_group_is_rejected(self):
        with self.assertRaisesRegex(
                tracklets.TrackletContractError, "duplicate keyframe"):
            tracklets.epoch_fused_compat_v1(
                [observation(1, 10.0), observation(1, 11.0)], PARAMS)


class BuildMeasurementsTest(unittest.TestCase):
    def setUp(self):
        self.tracks = {
            1: make_track(
                1, 0, {
                    keyframe: centred_box(PANO_W / 2)
                    for keyframe in range(6)
                }),
            2: make_track(
                2, 0, {
                    keyframe: centred_box(0.75 * PANO_W)
                    for keyframe in range(3)
                }),
            3: make_track(3, 0, {0: centred_box(1000)}),
        }

    def test_audit_membership_is_gate_and_output_ids_are_globally_scoped(self):
        audits = {
            1: audit_for(self.tracks[1]),
            2: audit_for(self.tracks[2]),
        }
        measurements = tracklets.build_measurements(
            self.tracks, audits, PANO_W, PARAMS)
        ids = {measurement.tracklet_id for measurement in measurements}
        self.assertEqual(len(ids), 2)
        self.assertTrue(any(value.endswith("#T1") for value in ids))
        self.assertTrue(any(value.endswith("#T2") for value in ids))
        self.assertTrue(all("@sha256:" in value for value in ids))

    def test_drop_is_excluded_and_partial_segments_reach_reducer(self):
        audits = {
            1: audit_for(self.tracks[1], "drop", []),
            2: audit_for(
                self.tracks[2], "keep_partial",
                [{"start_t": 0, "end_t": 1}]),
        }
        measurements = tracklets.build_measurements(
            self.tracks, audits, PANO_W, PARAMS)
        self.assertEqual(len(measurements), 1)
        self.assertTrue(measurements[0].tracklet_id.endswith("#T2"))
        self.assertEqual(measurements[0].anchor_keyframe_idx, 1)
        self.assertAlmostEqual(
            measurements[0].bearing_camera_cw_deg,
            geo.azimuth_of_pano_column(0.75 * PANO_W, PANO_W))

    def test_measurements_are_sorted(self):
        audits = {
            1: audit_for(self.tracks[1]),
            2: audit_for(self.tracks[2]),
        }
        measurements = tracklets.build_measurements(
            self.tracks, audits, PANO_W, PARAMS)
        keys = [
            (measurement.anchor_keyframe_idx, measurement.tracklet_id)
            for measurement in measurements
        ]
        self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main()
