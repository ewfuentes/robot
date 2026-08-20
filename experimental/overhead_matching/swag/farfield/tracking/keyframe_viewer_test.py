import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.tracking import (
    keyframe_viewer as kv,
    track_builder as tb,
)

PANO_W = 7680


def make_artifact(range_name, tracks, config=None, rejected=None):
    """A tracks_*.json-shaped dict, in the tracklets_test fixture style."""
    cfg = config or tb.TrackBuilderConfig(reference_pano_width=PANO_W)
    return {
        "range": {"name": range_name, "k_start": 0, "k_end": 20},
        "config": dataclasses.asdict(cfg),
        "tracks": tracks,
        "rejected_births": rejected or [],
        "track_overlaps": [],
    }


def make_track(track_id, birth_obs_id, records):
    return {
        "track_id": track_id,
        "birth_obs_id": birth_obs_id,
        "birth_keyframe": records[0]["keyframe"] if records else 0,
        "status": "closed", "close_reason": "starved",
        "end_keyframe": None, "last_keyframe": None,
        "modal_label": "man_made=tower", "n_supported_keyframes": 0,
        "records": records,
    }


def record(keyframe, supports=(), mask_bbox=None, origin=(100.0, 200)):
    return {
        "keyframe": keyframe, "action": "continue_mask",
        "window_origin": list(origin), "window_px": 1024,
        "mask_area": 100, "mask_bbox_window": mask_bbox,
        "supports": list(supports),
    }


def support(obs_id, iou, iom, iob):
    return {"obs_id": obs_id, "class": "recorded-at-run-time",
            "box_window": [0, 0, 10, 10], "iou": iou,
            "inter_over_mask": iom, "inter_over_box": iob}


class LoadTrackArtifactsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, filename, artifact):
        (self.run_dir / filename).write_text(json.dumps(artifact))

    def test_empty_run_dir_is_a_pointed_error(self):
        with self.assertRaises(SystemExit) as ctx:
            kv.load_track_artifacts(self.run_dir)
        self.assertIn(str(self.run_dir), str(ctx.exception))
        self.assertIn("tracks_*.json", str(ctx.exception))

    def test_loads_every_range_file(self):
        # The old viewer took next(glob) -- the FIRST file only -- silently
        # dropping every other range.
        self.write("tracks_a.json", make_artifact("legA", []))
        self.write("tracks_b.json", make_artifact("legB", []))
        self.assertEqual(set(kv.load_track_artifacts(self.run_dir)),
                         {"legA", "legB"})

    def test_range_name_comes_from_the_record_not_a_default(self):
        # The old viewer defaulted a missing name to the literal "full_leg1"
        # and aimed every link at pages that never existed. A missing name is
        # an error now.
        artifact = make_artifact("x", [])
        del artifact["range"]["name"]
        self.write("tracks_full_leg1.json", artifact)
        with self.assertRaises(SystemExit) as ctx:
            kv.load_track_artifacts(self.run_dir)
        self.assertIn("range name", str(ctx.exception))

    def test_filename_never_supplies_the_range_name(self):
        # Names key off the artifact record even when the filename disagrees.
        self.write("tracks_misnamed.json", make_artifact("real_name", []))
        self.assertEqual(list(kv.load_track_artifacts(self.run_dir)),
                         ["real_name"])

    def test_duplicate_range_names_are_refused(self):
        self.write("tracks_a.json", make_artifact("leg", []))
        self.write("tracks_b.json", make_artifact("leg", []))
        with self.assertRaises(SystemExit) as ctx:
            kv.load_track_artifacts(self.run_dir)
        self.assertIn("leg", str(ctx.exception))


class RecordedConfigTest(unittest.TestCase):
    def test_reconstructs_the_recorded_dataclass(self):
        cfg = tb.TrackBuilderConfig(reference_pano_width=PANO_W,
                                    clean_iou=0.30)
        artifact = make_artifact("leg", [], config=cfg)
        self.assertEqual(kv.recorded_config(artifact), cfg)

    def test_missing_config_is_an_error_not_a_default(self):
        artifact = make_artifact("leg", [])
        del artifact["config"]
        with self.assertRaises(SystemExit):
            kv.recorded_config(artifact)


class TrackAssociationsTest(unittest.TestCase):
    def test_supports_reclassified_under_the_recorded_config(self):
        # Recorded run used clean_iou=0.30: iou 0.35/iom 0.9 is a clean
        # continuation THERE, while today's default (0.45) would demote it to
        # weak. The viewer must show the run as it was built.
        recorded = tb.TrackBuilderConfig(reference_pano_width=PANO_W,
                                         clean_iou=0.30)
        track = make_track(0, "seed_obs", [
            record(5, supports=[support("f0005__lm1__box0",
                                        iou=0.35, iom=0.9, iob=0.5)]),
        ])
        artifacts = {"leg": make_artifact("leg", [track], config=recorded)}
        by_obs, _, _, _ = kv.track_associations(artifacts)
        self.assertEqual(by_obs[(5, "f0005__lm1__box0")],
                         [("leg_T0", "continue_clean")])
        # Sanity: a fresh default config would have said "weak" -- the value
        # the old viewer (which built TrackBuilderConfig()) would have shown.
        self.assertEqual(
            tb.classify_support(
                {"iou": 0.35, "inter_over_mask": 0.9, "inter_over_box": 0.5},
                tb.TrackBuilderConfig(reference_pano_width=PANO_W)),
            "weak")

    def test_each_range_classifies_under_its_own_config(self):
        loose = tb.TrackBuilderConfig(reference_pano_width=PANO_W,
                                      clean_iou=0.30)
        strict = tb.TrackBuilderConfig(reference_pano_width=PANO_W)
        sup = support("obs", iou=0.35, iom=0.9, iob=0.5)
        artifacts = {
            "loose": make_artifact(
                "loose", [make_track(0, "a", [record(1, supports=[sup])])],
                config=loose),
            "strict": make_artifact(
                "strict", [make_track(0, "b", [record(1, supports=[sup])])],
                config=strict),
        }
        by_obs, _, _, _ = kv.track_associations(artifacts)
        self.assertEqual(sorted(by_obs[(1, "obs")]),
                         [("loose_T0", "continue_clean"),
                          ("strict_T0", "weak")])

    def test_track_keys_carry_the_range_name(self):
        # Two ranges may reuse track_id 0; keys must not collide.
        artifacts = {
            "legA": make_artifact(
                "legA", [make_track(0, "obsA",
                                    [record(2, mask_bbox=[10, 20, 30, 40])])]),
            "legB": make_artifact(
                "legB", [make_track(0, "obsB",
                                    [record(2, mask_bbox=[1, 2, 3, 4])])]),
        }
        _, masks, seeded, _ = kv.track_associations(artifacts)
        self.assertEqual({key for key, _, _ in masks[2]},
                         {"legA_T0", "legB_T0"})
        self.assertEqual(seeded["obsA"], ["legA_T0"])
        self.assertEqual(seeded["obsB"], ["legB_T0"])

    def test_mask_boxes_shift_by_window_origin(self):
        track = make_track(0, "obs", [
            record(3, mask_bbox=[10, 20, 30, 40], origin=(100.0, 200)),
        ])
        _, masks, _, _ = kv.track_associations(
            {"leg": make_artifact("leg", [track])})
        self.assertEqual(masks[3], [("leg_T0", "continue_mask",
                                     (110.0, 220, 130.0, 240))])

    def test_rejected_births_surface_their_health(self):
        artifact = make_artifact(
            "leg", [], rejected=[{"obs_id": "bad_obs", "keyframe": 0,
                                  "health": {"ok": False,
                                             "reason": "fragmented"}}])
        _, _, _, rejected = kv.track_associations({"leg": artifact})
        self.assertEqual(rejected["bad_obs"]["reason"], "fragmented")


if __name__ == "__main__":
    unittest.main()
