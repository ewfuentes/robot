
import unittest

import experimental.beacon_sim.belief_road_map_planner_python as brm
import experimental.beacon_sim.test_helpers_python as helpers
import experimental.beacon_sim.ekf_slam_python as esp

class BeliefRoadMapPlannerPythonTest(unittest.TestCase):
    def test_belief_road_map_planner(self):
        # Setup
        ekf_config = esp.EkfSlamConfig(
            max_num_beacons=11,
            initial_beacon_uncertainty_m=100.0,
            along_track_process_noise_m_per_rt_meter=0.05,
            cross_track_process_noise_m_per_rt_meter=0.05,
            pos_process_noise_m_per_rt_s=0.0,
            heading_process_noise_rad_per_rt_meter=1e-3,
            heading_process_noise_rad_per_rt_s=0.0,
            beacon_pos_process_noise_m_per_rt_s=1e-6,
            range_measurement_noise_m=1e-1,
            bearing_measurement_noise_rad=1e-1,
            on_map_load_position_uncertainty_m=2.0,
            on_map_load_heading_uncertainty_rad=0.5,
        )

        brm_options = brm.BeliefRoadMapOptions(
            max_sensor_range_m=3.0,
            uncertainty_tolerance=None,
            max_num_edge_transforms=1000,
            timeout=None)

        P_LONE_BEACON = 0.5
        P_STACKED_BEACON = 0.1
        P_NO_STACK_BEACON = 0.01
        road_map, ekf, potential = helpers.create_diamond_environment(
            ekf_config, P_LONE_BEACON, P_NO_STACK_BEACON, P_STACKED_BEACON
        )

        # Action
        plan = brm.compute_belief_road_map_plan(road_map, ekf, potential, brm_options)

        # Verification
        self.assertIsNotNone(plan)
        self.assertNotIn(1, plan.nodes)



if __name__ == "__main__":
    unittest.main()
