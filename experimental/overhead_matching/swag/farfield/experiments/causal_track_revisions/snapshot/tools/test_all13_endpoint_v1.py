import copy
import json
import unittest
from pathlib import Path
from all13_filter_overlay_v1 import prepare_releases
from live_prefix_filter_overlay import prepare_releases as original
from experimental.overhead_matching.swag.farfield.localization import odometry_profiles
O=Path('/data/farfield_matching/runs/260913_accuracy_recovery')
class EndpointTest(unittest.TestCase):
    def test_existing_prefix_exact_and_tail_does_not_flush(self):
        ev=json.loads((O/'portland1_track_revisions_v1.end180.json').read_text())
        base=Path('/data/farfield_matching/artifacts/localization_inputs/portland_flight_20260906_leg1/promatch_20260912_dedup_epson_v1')
        ts=odometry_profiles.load_timestamps(base,492)
        self.assertEqual(original(ev,ts,180),prepare_releases(ev,ts,180))
        available=copy.deepcopy(ev['emissions'][0]);available['release_keyframe']=491;available['available_time_s']=ts[-1]
        future=copy.deepcopy(available);future['available_time_s']=ts[-1]+0.01
        releases,diagnostics=prepare_releases({'emissions':[available,future]},ts,491)
        self.assertEqual(len(releases),1)
        self.assertEqual(releases[0].release_keyframe_idx,491)
        self.assertEqual(diagnostics[0]['available_time_s'],ts[-1])
        available['available_time_s']=ts[-1]-0.01
        with self.assertRaises(ValueError):prepare_releases({'emissions':[available]},ts,491)
if __name__=='__main__':unittest.main()
