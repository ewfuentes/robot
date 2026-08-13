import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    filter as pf,
    run_log,
    scenario,
    structs,
)


class RunLogRoundTripTest(unittest.TestCase):
    def test_round_trip(self):
        cfg = scenario.straight_leg(speed_mps=20.0, keyframe_period_s=5.0,
                                    epoch_length_keyframes=3)
        data = scenario.generate(cfg)
        filter_config = structs.FilterConfig(
            n_particles=200, seed=11,
            init=structs.GaussianInit(data.truth[0].east_m,
                                      data.truth[0].north_m, 300.0),
            checkpoint_every=5)
        history = pf.run_filter(filter_config, data.catalog, data.odometry,
                                data.measurements, data.tables)
        manifest = structs.RunManifest(
            schema_version=structs.SCHEMA_VERSION,
            scenario_name=cfg.name,
            anchor_lat_deg=cfg.anchor_lat_deg,
            anchor_lon_deg=cfg.anchor_lon_deg,
            n_keyframes=data.n_keyframes,
            filter_config=filter_config,
            landmarks=cfg.landmarks,
            matcher_version=scenario.MATCHER_VERSION,
            particle_history_sha256=history.particle_history_sha256)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_log.write_run(run_dir, manifest, data.truth, data.odometry,
                              data.measurements, data.tables, history)
            loaded = run_log.read_run(run_dir)

        self.assertEqual(loaded.manifest, manifest)
        self.assertEqual(loaded.manifest.schema_version,
                         structs.SCHEMA_VERSION)
        self.assertEqual(loaded.truth, data.truth)
        self.assertEqual(loaded.odometry, data.odometry)
        self.assertEqual(loaded.measurements, data.measurements)
        self.assertEqual(loaded.tables, data.tables)
        self.assertEqual(loaded.health, history.health)
        self.assertEqual(loaded.proposal_events, history.proposal_events)
        self.assertEqual(set(loaded.checkpoints.keys()),
                         set(history.checkpoints.keys()))
        for kf, belief in history.checkpoints.items():
            np.testing.assert_array_equal(
                loaded.checkpoints[kf]["east_m"], belief.east_m)
            np.testing.assert_array_equal(
                loaded.checkpoints[kf]["log_weight"], belief.log_weight)
            np.testing.assert_array_equal(
                loaded.checkpoints[kf]["proposal_event_id"],
                belief.proposal_event_id)

    def test_health_records_survive_association_payload(self):
        record = structs.HealthRecord(
            keyframe_idx=3, ess=42.0, resampled=True, mean_east_m=1.0,
            mean_north_m=2.0, mean_heading_deg=3.0, position_std_m=4.0,
            map_east_m=1.0,
            map_north_m=2.0,
            map_heading_deg=3.0,
            heading_std_deg=1.5,
            n_measurements=1,
            associations=[structs.AssociationPosterior(
                tracklet_id="trk_a", anchor_keyframe_idx=3, null_share=0.1,
                responsibilities={"lm_a": 0.9})])
        import msgspec
        encoded = msgspec.json.encode(record)
        decoded = msgspec.json.decode(encoded, type=structs.HealthRecord)
        self.assertEqual(decoded, record)


if __name__ == "__main__":
    unittest.main()
