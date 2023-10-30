#include "experimental/beacon_sim/ethans_super_cool_planner.hh"

#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"
#include "planning/probabilistic_road_map.hh"

#include <iostream>

namespace robot::experimental::beacon_sim {

using SuccessorFunc = std::function<std::vector<planning::Successor<int>>(const int)>;

SuccessorFunc make_successor_function(const planning::RoadMap &map) {
    return [&map](const int parent) {
        std::vector<planning::Successor<int>> successors;
        for (const auto &[neighbor_idx, neighbor_pos] : map.neighbors(parent)) {
            successors.push_back({
                .state = neighbor_idx,
                .edge_cost = (neighbor_pos - map.point(parent)).norm(),
            });
        }
        return successors;
    };
}

using TerminateRolloutFunc = std::function<bool(const Candidate &, int)>;

TerminateRolloutFunc make_terminate_func(const int max_steps) {
    return [max_steps]([[maybe_unused]] const Candidate &candidate, const int num_steps) {
        return num_steps >= max_steps;
    };
}

std::ostream& operator<<(std::ostream& os, const Candidate& candidate) {
    os << "Candidate{"
       << "belief_trace=" << candidate.belief.cov_in_robot.trace() 
       << ", path_history=[";
    for (const auto& node : candidate.path_history) {
        os << node << ", ";
    }
    os << "]}";
    return os;
}

TEST(EthansSuperCoolPlannerTest, RolloutHappyCase) {
    // Setup

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

    const auto &[road_map, ekf_slam, potential] = create_grid_environment(ekf_config, 0.5);


    Candidate candidate = {
        .belief = ekf_slam.estimate().robot_belief(),
        .path_history = {road_map.START_IDX},
    };

    RollOutArgs roll_out_args = {
        .num_roll_outs = 1000,
    };

    constexpr double max_sensor_range_m = 3.0;

    planning::BeliefUpdater<RobotBelief> belief_updater =
        make_belief_updater(road_map, 
                            max_sensor_range_m,
                            ekf_slam,
                            std::nullopt,
                            TransformType::INFORMATION);

    const SuccessorFunc successor_func = make_successor_function(road_map);
    const TerminateRolloutFunc terminate_rollout_func = make_terminate_func(10);

    // Action

    auto candidates = rollout(road_map, 
                              terminate_rollout_func, 
                              candidate,
                              successor_func, 
                              belief_updater, 
                              roll_out_args);

    std::cout << "from initial candidate: " << candidate << std::endl;
    for (const auto &candidate : candidates) {
        std::cout << "\t candidate: " << candidate << std::endl;
    }
    // Verification
    EXPECT_TRUE(candidates.size() == roll_out_args.num_roll_outs);
}

}  // namespace robot::experimental::beacon_sim
