#include "experimental/beacon_sim/ethans_super_cool_planner.hh"

#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {

auto make_grid_successor(const planning::RoadMap &road_map,
                         const Eigen::Vector2d &start_in_world,
                         const Eigen::Vector2d &goal_in_world,
                         const int start_goal_connection_radius_m) {
    return [&road_map, start_goal_connection_radius_m, START_IDX, GOAL_IDX](const int node_idx) -> std::vector<planning::Successor<int>> {
        std::vector<planning::Successor<int>> out;
        if (node_idx == START_IDX || node_idx == GOAL_IDX) {
            for (int i = 0; i < static_cast<int>(road_map.points.size()); i++) {
                const Eigen::Vector2d &pt_in_local = road_map.points.at(i);
                const double dist_m = (pt_in_local - start_in_local).norm();
                if (dist_m < start_goal_connection_radius_m) {
                    out.push_back({.state = i, .edge_cost = dist_m});
                }
            }
        } else {
            const Eigen::Vector2d &curr_pt_in_local = road_map.points.at(node_idx);
            for (int i = 0; i < static_cast<int>(road_map.points.size()); i++) {
                if (road_map.adj(i, node_idx)) {
                    const Eigen::Vector2d &other_in_local = road_map.points.at(i);
                    const double dist_m = (curr_pt_in_local - other_in_local).norm();
                    out.push_back({.state = i, .edge_cost = dist_m});
                }
            }

            const double dist_to_goal_m = (curr_pt_in_local - goal_in_local).norm();
            if (dist_to_goal_m < start_goal_connection_radius_m) {
                out.push_back({.state = GOAL_IDX, .edge_cost = dist_to_goal_m});
            }
        }
        return out;
    };
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
        .path_history = {START_NODE_INDEX},
    };

    RollOutArgs roll_out_args = {

    };

    const Eigen::Vector2d GOAL_STATE = {10, -5};

    planning::BeliefUpdater<RobotBelief> belief_updater =
        make_belief_updater(road_map, GOAL_STATE, 3.0, ekf_slam, {GRID_BEACON_ID}, TransformType::INFORMATION);

    // Action

    auto candidates = rollout(road_map, candidate, , belief_updater, roll_out_args);

    // Verification
    EXPECT_TRUE(false);
}

}  // namespace robot::experimental::beacon_sim
