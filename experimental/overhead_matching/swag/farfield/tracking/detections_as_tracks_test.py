import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    dataset as dataset_lib,
    testing,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    detections_as_tracks as dat,
    tracklets,
)

PANO_W, PANO_H = 256, 128
FOV = 90.0


def ingest(tmp: Path):
    base = testing.make_dataset(tmp / "ds", n_frames=2,
                                pano_size=(PANO_W, PANO_H))
    stems = sorted(p.stem for p in (base / "panorama").glob("*.jpg"))
    seam = testing.landmark(
        "Graves Light", [(0, 950, 300, 1000, 500), (270, 0, 300, 60, 500)],
        primary=("man_made", "lighthouse"))
    seam["additional_tags"] += [
        {"key": "distance_estimate", "value": "2km_to_10km"},
        {"key": "colour", "value": "white"}]
    plain = testing.landmark("Tower", [(90, 100, 200, 300, 600)])
    plain["additional_tags"] = []
    per_stem = {stems[0]: [seam, plain], stems[1]: [plain]}
    testing.make_predictions(tmp / "fl", per_stem, dataset_name=base.name)
    return dataset_lib.run_ingest(
        base, tmp / "fl", dataset_lib.IngestParams(FOV, 25.0, 0.3))


class DetectionsAsTracksTest(unittest.TestCase):
    def test_tracks_and_audits_reproduce_each_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ingest(Path(tmp))
        observations = sorted(result.observations,
                              key=lambda o: (o.frame_idx, o.landmark_idx))
        self.assertEqual(len(observations), 3)
        self.assertTrue(observations[0].seam_merged)

        tracks, audits = {}, {}
        for index, obs in enumerate(observations):
            tracks[index] = dat.detection_track(
                index, obs, pano_w=PANO_W, pano_h=PANO_H, fov_deg=FOV)
            audits[index] = dat.passthrough_audit(obs)

        # The audit restates the detection: identity tags in order, the name
        # as the only candidate, distance dropped, nothing decided.
        tags = [t["tag"] for t in audits[0]["primary_object"]["tags"]]
        self.assertEqual(tags, ["man_made=lighthouse", "colour=white"])
        self.assertEqual(
            [c["name"] for c in audits[0]["primary_object"]["name_candidates"]],
            ["Graves Light"])
        self.assertEqual(audits[1]["primary_object"]["name_candidates"], [])
        self.assertEqual(audits[0]["verdict"], "keep")

        accepted = tracklets.build_accepted_tracklets(tracks, audits)
        self.assertEqual(len(accepted), 3)
        bearings = tracklets.build_camera_bearing_observations(
            accepted, PANO_W, 1.0)
        self.assertEqual(len(bearings), 3)
        for item, obs in zip(
                sorted(bearings, key=lambda b: b.tracklet_id),
                sorted(observations,
                       key=lambda o: tracklets.tracklet_id(
                           {"track_id": observations.index(o)}))):
            self.assertEqual(item.keyframe_idx, obs.frame_idx)
            # Box midpoint vs the ingest's edge-derived bearing: one pano
            # pixel is 360/256 deg, seam boxes round on both faces.
            self.assertLess(abs(item.bearing_camera_cw_deg
                                - obs.bearing_camera_cw_deg) % 360.0, 3.0)
            self.assertLess(
                abs(item.angular_width_deg - obs.angular_width_deg), 3.0)

    def test_landmark_kind_is_not_guessed(self):
        # The field is required by the schema; the detection never judged it.
        with tempfile.TemporaryDirectory() as tmp:
            result = ingest(Path(tmp))
        for obs in result.observations:
            self.assertEqual(
                dat.passthrough_audit(obs)["landmark_kind"], "mixed_or_unclear")


if __name__ == "__main__":
    unittest.main()
