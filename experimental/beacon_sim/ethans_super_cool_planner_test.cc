#include "gtest/gtest.h"


#include "experimental/beacon_sim/ethans_super_cool_planner.hh"

#include "experimental/beacon_sim/test_helpers.hh"

namespace robot::experimental::beacon_sim {

TEST(EthansSuperCoolPlannerTest, RolloutHappyCase){
    //Setup



    const EkfSlamConfig ekf_config{
        .max_num_beacons = 1,
        .initial_beacon_uncertainty_m = 100.0,
        .along_track_process_noise_m_per_rt_meter = 0.05,
        .cross_track_process_noise_m_per_rt_meter = 0.05,
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 1e-3,
        .heading_process_noise_rad_per_rt_s = 0.0,
        .beacon_pos_process_noise_m_per_rt_s = 1e-6,
        .range_measurement_noise_m = 1e-1,
        .bearing_measurement_noise_rad = 1e-1,
        .on_map_load_position_uncertainty_m = 2.0,
        .on_map_load_heading_uncertainty_rad = 0.5,
    };
    
    const auto &[road_map, ekf_slam, potential] = create_grid_environment(
        ekf_config, 0.5
    );
    

    Candidate candidate = {
        .belief = ekf_slam.estimate().robot_belief(),
        .path_history = { START_NODE_INDEX },
    };

    RollOutArgs roll_out_args = {

    };

    const Eigen::Vector2d GOAL_STATE = {10, -5};


    planning::BeliefUpdater<RobotBelief> belief_updater = make_belief_updater(road_map, GOAL_STATE, 3.0, ekf_slam, {GRID_BEACON_ID});

    

    //Action

    auto candidates = rollout(
        road_map, candidate, belief_updater, roll_out_args
    );

    //Verification
    EXPECT_TRUE(false);


}



} // namespace