import dataclasses
import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield.tracking import (
    track_builder as tb,
)

PANO_W, PANO_H = 7680, 3840
WIN = 256


def cfg(**overrides):
    base = tb.TrackBuilderConfig(reference_pano_width=PANO_W, window_px=WIN,
                                 patience_keyframes=3, drift_patience=2,
                                 min_mask_area_px=10)
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def default_cfg():
    """The stage's defaults, with only the mandatory pano width supplied."""
    return tb.TrackBuilderConfig(reference_pano_width=PANO_W)


class FakeObs:
    def __init__(self, obs_id, tag=("man_made", "tower"), name=None):
        self.obs_id = obs_id
        self.primary_tag_key, self.primary_tag_value = tag
        self.additional_tags = [["name", name]] if name else []
        self.confidence = "high"
        self.landmark_idx = 0
        self.description = ""


def rect_mask(x0, y0, x1, y1, size=WIN):
    m = np.zeros((size, size), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class FakeBackend:
    """Returns a scripted mask (window coords) for every frame of each call,
    in call order."""

    def __init__(self, masks_per_call):
        self.masks_per_call = list(masks_per_call)
        self.calls = []

    def propagate(self, frames, prompt_box=None, prompt_mask=None):
        self.calls.append({"n": len(frames), "box": prompt_box,
                           "mask": prompt_mask,
                           "size": frames[0].shape[0]})
        mask = self.masks_per_call.pop(0)
        if callable(mask):
            mask = mask(frames[0].shape[0])
        return [mask.copy() for _ in frames]

    def propagate_batch(self, clips):
        """What the builder actually calls. Batching is a GPU-occupancy
        optimization in the real backend, not a semantic one, so the fake
        simply serves each clip in order -- which is also what keeps
        `masks_per_call` scripted per track, as these tests expect."""
        return [self.propagate(frames, prompt_box=box, prompt_mask=mask)
                for frames, box, mask in clips]


def crops_fn_factory(builder):
    def crops_fn(track, size):
        x0, y0 = builder.window_origin(track, size)
        y0 = min(max(y0, 0), PANO_H - size)
        frames = [np.zeros((size, size, 3), np.uint8) for _ in range(3)]
        origins = [(x0, int(y0))] * 3
        return frames, origins
    return crops_fn


def centered_pano_box(center_x, center_y, w, h):
    return [center_x - w / 2, center_y - h / 2,
            center_x + w / 2, center_y + h / 2]


class ConfigContractTest(unittest.TestCase):
    def test_reference_pano_width_is_mandatory(self):
        # Audit finding: the pixel thresholds are tuned at 8K-pano resolution
        # and are resolution-ABSOLUTE. The width they were applied to must be
        # recorded, so the field has no default.
        with self.assertRaises(TypeError):
            tb.TrackBuilderConfig()

    def test_recorded_config_round_trips(self):
        # Readers reconstruct the recorded config with
        # TrackBuilderConfig(**stored); asdict must therefore carry every
        # field, including reference_pano_width.
        stored = dataclasses.asdict(cfg())
        self.assertEqual(stored["reference_pano_width"], PANO_W)
        self.assertEqual(tb.TrackBuilderConfig(**stored), cfg())


class ClassifySupportTest(unittest.TestCase):
    """Thresholds anchored on real M2 measurements."""

    def check(self, iou, iom, iob, expected):
        got = tb.classify_support(
            {"iou": iou, "inter_over_mask": iom, "inter_over_box": iob},
            default_cfg())
        self.assertEqual(got, expected)

    def test_m2_cases(self):
        self.check(0.54, 0.79, 0.63, "continue_clean")   # CHT
        self.check(0.64, 0.99, 0.64, "continue_clean")   # crane group
        self.check(0.48, 0.87, 0.52, "continue_clean")   # monument
        self.check(0.30, 0.95, 0.30, "merge_superset")   # fort, wider box
        self.check(0.22, 0.91, 0.23, "merge_superset")   # bridge, wider box
        self.check(0.10, 1.00, 0.10, "merge_superset")   # tower in crane box
        self.check(0.05, 0.16, 0.07, "none")             # drifted lattice mask
        self.check(0.00, 0.00, 0.00, "none")
        # Split-shaped: detection inside a larger mask.
        self.check(0.20, 0.25, 0.90, "split_child")

    def test_speck_in_giant_box_is_context_not_support(self):
        # Regression: T172 - a dying island-remnant mask was kept alive 50
        # keyframes by successive OTHER islands' giant boxes (im=1.0,
        # ib=0.01-0.08). Below the coherence floor -> context.
        self.check(0.01, 1.00, 0.01, "context")
        self.check(0.04, 1.00, 0.04, "context")
        self.check(0.08, 0.95, 0.08, "context")
        # Genuine granularity supersets keep their class (fort: ib >= 0.16).
        self.check(0.30, 0.95, 0.30, "merge_superset")
        self.check(0.16, 0.91, 0.16, "merge_superset")

    def test_context_detection_neither_sustains_nor_is_claimed(self):
        speck = rect_mask(120, 120, 150, 150)  # 30x30 mask
        backend = FakeBackend([speck, speck])
        builder = tb.TrackBuilder(backend, cfg(), PANO_W, PANO_H)
        obs0 = FakeObs("f0000__lm0__box0")
        pano_box = centered_pano_box(1007, 1927, 34, 34)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        giant = FakeObs("f0001__lm9__box0", tag=("place", "island"))
        # 800x200 box containing the speck: im=1.0, ib ~ 0.006 -> context.
        builder.step(0, crops_fn_factory(builder), [giant],
                     {giant.obs_id: centered_pano_box(1000, 1920, 800, 200)})
        track = builder.tracks[0]
        rec = track.records[-1]
        self.assertEqual(rec["supports"][0]["class"], "context")
        self.assertEqual(rec["action"], "unsupported")
        self.assertEqual(track.unsupported_streak, 1)
        # Fix 2: the giant context box must not inflate the window.
        self.assertEqual(track.window_px, 256)
        # The giant detection is unclaimed and seeds its own track.
        self.assertEqual(len(builder.tracks), 2)
        self.assertEqual(builder.tracks[1].birth_obs_id, giant.obs_id)

    def test_occluder_box_grazing_mask_is_not_support(self):
        # Regression: T185/f0268 - a foreground island's giant box covered
        # 0.52 of a background island's mask with iou 0.00. Containment-only
        # weak support needs mutual agreement (complement floor).
        self.check(0.00, 0.52, 0.00, "none")
        self.check(0.02, 0.65, 0.02, "none")   # shared giant box at f0167
        # Legitimate partial overlaps keep their weak class.
        self.check(0.04, 0.60, 0.30, "weak")
        self.check(0.16, 0.30, 0.30, "weak")

    def test_containment_guard_blocks_truncating_reanchor(self):
        # Regression: T0/f0011 - displaced box cleared clean_iou against a
        # spread mask but covered only 0.63 of it; re-anchoring stole the
        # track onto the neighboring building. Must demote to weak.
        self.check(0.46, 0.63, 0.63, "weak")
        # Healthy re-anchors keep working (M2 measurements).
        self.check(0.54, 0.79, 0.63, "continue_clean")


class MaskHealthTest(unittest.TestCase):
    def test_clean_mask_passes(self):
        mask = rect_mask(100, 100, 140, 200)
        health = tb.mask_health(mask, [95, 95, 145, 205], default_cfg())
        self.assertTrue(health["ok"])

    def test_fragmented_mask_fails(self):
        mask = np.zeros((WIN, WIN), dtype=bool)
        rng = np.random.RandomState(0)
        for _ in range(25):  # lattice-tower style confetti
            x, y = rng.randint(20, 220, 2)
            mask[y:y + 4, x:x + 4] = True
        health = tb.mask_health(mask, [20, 20, 224, 224], default_cfg())
        self.assertFalse(health["ok"])
        self.assertEqual(health["reason"], "fragmented")
        # Regression: numpy scalars must not leak into the record
        # (np.bool_ crashed artifact serialization on r001/f0149).
        import json
        json.dumps(health)

    def test_spilled_mask_fails(self):
        mask = rect_mask(0, 0, 200, 200)
        health = tb.mask_health(mask, [150, 150, 200, 200], default_cfg())
        self.assertFalse(health["ok"])


class TrackBuilderTest(unittest.TestCase):
    def _mk(self, masks):
        backend = FakeBackend(masks)
        builder = tb.TrackBuilder(backend, cfg(), PANO_W, PANO_H)
        return backend, builder

    def test_birth_gate_rejects_fragmented_mask(self):
        confetti = np.zeros((WIN, WIN), dtype=bool)
        rng = np.random.RandomState(1)
        for _ in range(30):
            x, y = rng.randint(10, 240, 2)
            confetti[y:y + 3, x:x + 3] = True
        _, builder = self._mk([confetti])
        obs = FakeObs("f0000__lm0__box0")
        builder.seed_unassigned(
            0, [obs], {obs.obs_id: centered_pano_box(1000, 1920, 200, 200)})
        builder.step(0, crops_fn_factory(builder), [], {})
        self.assertEqual(builder.tracks[0].status, "closed")
        self.assertIn("birth_", builder.tracks[0].close_reason)
        self.assertEqual(len(builder.rejected_births), 1)

    def test_birth_record_serializes_mask_geometry(self):
        good = rect_mask(108, 88, 148, 168)
        _, builder = self._mk([good])
        obs = FakeObs("f0000__lm0__box0")
        builder.seed_unassigned(
            0, [obs], {obs.obs_id: centered_pano_box(1000, 1920, 40, 80)})
        builder.step(0, crops_fn_factory(builder), [], {})
        birth = builder.tracks[0].records[0]
        self.assertEqual(birth["mask_area"], 40 * 80)
        self.assertEqual(birth["mask_bbox_window"], [108, 88, 147, 167])
        self.assertEqual(birth["supports"], [])

    def test_empty_initial_frame_still_seeds_next_keyframe(self):
        _, builder = self._mk([])
        obs = FakeObs("f0001__lm0__box0")
        box = centered_pano_box(1000, 1920, 40, 80)
        builder.step(0, crops_fn_factory(builder), [obs], {obs.obs_id: box})
        self.assertEqual(len(builder.tracks), 1)
        self.assertEqual(builder.tracks[0].birth_keyframe, 1)

    def test_terminal_step_supports_existing_but_does_not_seed_new_track(self):
        good = rect_mask(108, 88, 148, 168)
        _, builder = self._mk([good])
        first = FakeObs("f0000__lm0__box0")
        first_box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(
            0, [first], {first.obs_id: first_box})
        matching = FakeObs("f0001__lm0__box0")
        unclaimed = FakeObs(
            "f0001__lm1__box0", tag=("natural", "peak"))
        boxes = {
            matching.obs_id: first_box,
            unclaimed.obs_id: centered_pano_box(2000, 1920, 40, 80),
        }
        builder.step(
            0, crops_fn_factory(builder), [matching, unclaimed], boxes,
            allow_new_births=False)
        self.assertEqual(len(builder.tracks), 1)
        track = builder.tracks[0]
        self.assertEqual(track.records[-1]["action"], "reanchor_clean")
        supported = [
            support for support in track.records[-1]["supports"]
            if support["class"] in tb.SUPPORT_CLASSES
        ]
        self.assertEqual(
            [support["obs_id"] for support in supported], [matching.obs_id])
        self.assertNotEqual(track.birth_obs_id, unclaimed.obs_id)

    def test_all_unusable_tracks_still_seed_later_detection(self):
        good = rect_mask(108, 88, 148, 168)
        _, builder = self._mk([good])
        first = FakeObs("f0000__lm0__box0")
        box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(0, [first], {first.obs_id: box})
        builder.step(0, crops_fn_factory(builder), [], {})
        track = builder.tracks[0]
        track.prompt_box = None
        track.prompt_mask = np.zeros_like(good)
        track._mask_origin = track.last_origin

        later = FakeObs("f0002__lm0__box0")
        builder.step(1, crops_fn_factory(builder), [later],
                     {later.obs_id: box})
        self.assertEqual(track.close_reason, "mask_lost_in_window")
        self.assertEqual(len(builder.tracks), 2)
        self.assertEqual(builder.tracks[1].birth_obs_id, later.obs_id)
        self.assertEqual(builder.tracks[1].birth_keyframe, 2)

    def test_clean_track_reanchors_and_survives(self):
        good = rect_mask(108, 88, 148, 168)  # ~centered on window center
        _, builder = self._mk([good, good, good])
        obs0 = FakeObs("f0000__lm0__box0")
        pano_box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        for k in range(3):
            det = FakeObs(f"f{k + 1:04d}__lm0__box0")
            # Detection wherever the mask is: same pano box every keyframe.
            builder.step(k, crops_fn_factory(builder), [det],
                         {det.obs_id: pano_box})
        track = builder.tracks[0]
        self.assertEqual(track.status, "alive")
        actions = [r["action"] for r in track.records
                   if r["action"] != "birth"]
        self.assertTrue(all(a == "reanchor_clean" for a in actions), actions)
        self.assertEqual(track.end_keyframe, 3)
        # Supported detections must not seed duplicate tracks.
        self.assertEqual(len(builder.tracks), 1)

    def test_starvation_closes_after_patience(self):
        good = rect_mask(108, 88, 148, 168)
        _, builder = self._mk([good] * 6)
        obs0 = FakeObs("f0000__lm0__box0")
        pano_box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        for k in range(5):
            builder.step(k, crops_fn_factory(builder), [], {})
        track = builder.tracks[0]
        self.assertEqual(track.status, "closed")
        self.assertEqual(track.close_reason, "starved")
        # patience_keyframes=3 -> supported at birth k=0, closed at k=3.
        self.assertEqual(track.end_keyframe, 0)

    def test_drift_alarm_closes_early(self):
        good = rect_mask(108, 88, 148, 168)
        _, builder = self._mk([good] * 6)
        obs0 = FakeObs("f0000__lm0__box0", tag=("man_made", "tower"))
        pano_box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        for k in range(4):
            # Same-tag detection ~100 px away from the mask: near miss.
            det = FakeObs(f"f{k + 1:04d}__lmX__box0", tag=("man_made", "tower"))
            near_box = centered_pano_box(1000 + 115, 1920, 20, 60)
            builder.step(k, crops_fn_factory(builder), [det],
                         {det.obs_id: near_box})
            if builder.tracks[0].status == "closed":
                break
        track = builder.tracks[0]
        self.assertEqual(track.status, "closed")
        self.assertEqual(track.close_reason, "drift_alarm")
        # But the near-miss detections themselves seeded new tracks.
        self.assertGreater(len(builder.tracks), 1)

    def test_never_supported_track_uses_short_patience(self):
        good = rect_mask(108, 88, 148, 168)
        backend = FakeBackend([good] * 12)
        builder = tb.TrackBuilder(
            backend, cfg(patience_keyframes=10,
                         patience_unsupported_keyframes=2), PANO_W, PANO_H)
        obs0 = FakeObs("f0000__lm0__box0")
        pano_box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        for k in range(4):
            builder.step(k, crops_fn_factory(builder), [], {})
        track = builder.tracks[0]
        self.assertEqual(track.status, "closed")
        self.assertEqual(track.close_reason, "starved")
        self.assertEqual(track.last_keyframe, 2)  # closed at streak 2

    def test_supported_track_keeps_long_patience(self):
        good = rect_mask(108, 88, 148, 168)
        backend = FakeBackend([good] * 8)
        builder = tb.TrackBuilder(
            backend, cfg(patience_keyframes=10,
                         patience_unsupported_keyframes=2), PANO_W, PANO_H)
        obs0 = FakeObs("f0000__lm0__box0")
        pano_box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        det = FakeObs("f0001__lm0__box0")
        builder.step(0, crops_fn_factory(builder), [det],
                     {det.obs_id: pano_box})  # supported once
        for k in range(1, 7):
            builder.step(k, crops_fn_factory(builder), [], {})
        self.assertEqual(builder.tracks[0].status, "alive")

    def test_track_overlaps_recorded_for_coalive_pair(self):
        # Two tracks whose masks share pixels in pano space. Both masks sit
        # on their own founding box (so both pass birth gating); the boxes
        # are 20 px apart in pano space.
        m1 = rect_mask(100, 100, 160, 200)
        m2 = rect_mask(98, 100, 158, 200)
        backend = FakeBackend([m1, m2])
        builder = tb.TrackBuilder(backend, cfg(), PANO_W, PANO_H)
        obs_a = FakeObs("f0000__lm0__box0")
        obs_b = FakeObs("f0000__lm1__box0")
        boxes = {obs_a.obs_id: centered_pano_box(1000, 1920, 60, 100),
                 obs_b.obs_id: centered_pano_box(1020, 1920, 60, 100)}
        builder.seed_unassigned(0, [obs_a, obs_b], boxes)
        builder.step(0, crops_fn_factory(builder), [], {})
        self.assertEqual(len(builder.track_overlaps), 1)
        ov = builder.track_overlaps[0]
        self.assertEqual((ov["track_a"], ov["track_b"]), (0, 1))
        # Masks span pano x 972..1032 and 990..1050: 42px shared of 60px
        # width -> inter_over_min = 42/60.
        self.assertAlmostEqual(ov["inter_over_min"], 42 / 60, places=2)

    def test_mask_death_closes(self):
        good = rect_mask(108, 88, 148, 168)
        dead = np.zeros((WIN, WIN), dtype=bool)
        _, builder = self._mk([good, dead])
        obs0 = FakeObs("f0000__lm0__box0")
        pano_box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        builder.step(0, crops_fn_factory(builder), [], {})
        builder.step(1, crops_fn_factory(builder), [], {})
        self.assertEqual(builder.tracks[0].status, "closed")
        self.assertEqual(builder.tracks[0].close_reason, "mask_dead")

    def test_sustained_mid_track_fragmentation_closes(self):
        good = rect_mask(108, 88, 148, 168)
        # Two equal regions with a full gap: dominant component ~0.5 < 0.6.
        split = rect_mask(100, 88, 120, 168) | rect_mask(140, 88, 160, 168)
        _, builder = self._mk([good, split, split, split])
        obs0 = FakeObs("f0000__lm0__box0")
        pano_box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        for keyframe in range(4):
            det = FakeObs(f"f{keyframe + 1:04d}__lm0__box0")
            builder.step(keyframe, crops_fn_factory(builder), [det],
                         {det.obs_id: pano_box})
        track = builder.tracks[0]
        self.assertEqual(track.status, "closed")
        self.assertEqual(track.close_reason, "mask_fragmented")

    def test_brief_fragmentation_is_tolerated(self):
        good = rect_mask(108, 88, 148, 168)
        split = rect_mask(100, 88, 120, 168) | rect_mask(140, 88, 160, 168)
        _, builder = self._mk([good, split, good, split, good])
        obs0 = FakeObs("f0000__lm0__box0")
        pano_box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        for keyframe in range(5):
            det = FakeObs(f"f{keyframe + 1:04d}__lm0__box0")
            builder.step(keyframe, crops_fn_factory(builder), [det],
                         {det.obs_id: pano_box})
        self.assertEqual(builder.tracks[0].status, "alive")

    def test_window_grows_with_object_extent(self):
        # Wide mask centered in whatever window size is requested.
        def wide_mask(size):
            m = np.zeros((size, size), dtype=bool)
            m[size // 2 - 40:size // 2 + 40,
              size // 2 - 100:size // 2 + 100] = True
            return m
        backend = FakeBackend([wide_mask, wide_mask])
        builder = tb.TrackBuilder(backend, cfg(), PANO_W, PANO_H)
        obs0 = FakeObs("f0000__lm0__box0")
        # Birth box 700 px wide -> window 700*2=1400 -> quantized 1536.
        pano_box = centered_pano_box(1000, 1920, 700, 100)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        track = builder.tracks[0]
        self.assertEqual(track.window_px, 1536)
        builder.step(0, crops_fn_factory(builder), [], {})
        self.assertEqual(backend.calls[0]["size"], 1536)
        # Post-step extent = mask width 200 -> want 400 -> quantized 512.
        self.assertEqual(track.window_px, 512)

    def test_window_shrinks_back_for_small_extent(self):
        self.assertEqual(tb.window_size_for_extent(80, cfg()), 256)
        self.assertEqual(tb.window_size_for_extent(700, cfg()), 1536)
        self.assertEqual(
            tb.window_size_for_extent(5000, cfg()), cfg().window_max_px)

    def test_mask_translation_across_window_sizes(self):
        builder = tb.TrackBuilder(FakeBackend([]), cfg(), PANO_W, PANO_H)
        track = tb.Track(track_id=0, birth_obs_id="x", birth_keyframe=0)
        old = np.zeros((256, 256), dtype=bool)
        old[100:120, 200:240] = True
        track.prompt_mask = old
        track._mask_origin = (1000.0, 1800)
        # New window twice the size, origin shifted left/up by 128.
        out = builder._mask_in_window(track, (872.0, 1672), 512)
        self.assertEqual(int(out.sum()), int(old.sum()))
        ys, xs = np.nonzero(out)
        self.assertEqual((ys.min(), xs.min()), (100 + 128, 200 + 128))

    def test_superset_detection_supports_without_reanchor(self):
        good = rect_mask(108, 88, 148, 168)
        _, builder = self._mk([good, good])
        obs0 = FakeObs("f0000__lm0__box0")
        pano_box = centered_pano_box(1000, 1920, 40, 80)
        builder.seed_unassigned(0, [obs0], {obs0.obs_id: pano_box})
        wide = FakeObs("f0001__lm9__box0")
        # Much wider box containing the mask -> merge_superset.
        builder.step(0, crops_fn_factory(builder), [wide],
                     {wide.obs_id: centered_pano_box(1000, 1920, 400, 120)})
        track = builder.tracks[0]
        rec = track.records[-1]
        self.assertEqual(rec["action"], "continue_mask")
        self.assertEqual(rec["supports"][0]["class"], "merge_superset")
        self.assertEqual(track.unsupported_streak, 0)
        self.assertEqual(len(builder.tracks), 1)  # superset det doesn't seed


if __name__ == "__main__":
    unittest.main()
