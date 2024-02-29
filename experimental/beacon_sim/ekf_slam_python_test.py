
import unittest

from experimental.beacon_sim import ekf_slam_python as esp
from common.time import robot_time_python as rtp
from common.liegroups import se2_python as se2
from experimental.beacon_sim.generate_observations_python import BeaconObservation

import numpy as np
np.set_printoptions(linewidth=200)

class EkfSlamPythonTest(unittest.TestCase):
    def test_happy_case(self):
        # Setup
        config = esp.EkfSlamConfig(
            max_num_beacons=2,
            initial_beacon_uncertainty_m=100.0,
            along_track_process_noise_m_per_rt_meter=0.1,
            cross_track_process_noise_m_per_rt_meter=0.01,
            pos_process_noise_m_per_rt_s=0.01,
            heading_process_noise_rad_per_rt_meter=0.001,
            heading_process_noise_rad_per_rt_s=0.00001,
            beacon_pos_process_noise_m_per_rt_s=0.01,
            range_measurement_noise_m=0.05,
            bearing_measurement_noise_rad=0.005,
            on_map_load_position_uncertainty_m=0.1,
            on_map_load_heading_uncertainty_rad=0.01,
        )

        current_time = rtp.RobotTimestamp() + rtp.as_duration(123.456)
        dt = rtp.as_duration(0.5)
        ekf = esp.EkfSlam(config, current_time)
        BEACON_ID = 456
        BEACON_IN_LOCAL = np.array([2.0, 1.0])

        # Action
        old_robot_from_new_robot = se2.SE2(np.array([0.5, 0]))

        for i in range(10):
            current_time += dt
            ekf.predict(current_time, old_robot_from_new_robot)

            # Compute the observation
            beacon_in_robot = ekf.estimate().local_from_robot().inverse() * BEACON_IN_LOCAL

            range_m = np.linalg.norm(beacon_in_robot)
            bearing_rad = np.arctan2(beacon_in_robot[1], beacon_in_robot[0])

            obs = BeaconObservation(BEACON_ID, range_m, bearing_rad)

            ekf.update([obs])

        # Verification
        est_beacon_in_local = ekf.estimate().beacon_in_local(BEACON_ID)
        self.assertAlmostEqual(est_beacon_in_local[0], BEACON_IN_LOCAL[0])
        self.assertAlmostEqual(est_beacon_in_local[1], BEACON_IN_LOCAL[1])


if __name__ == "__main__":
    unittest.main()
