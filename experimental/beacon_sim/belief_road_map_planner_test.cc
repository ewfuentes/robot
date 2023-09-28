
#include "experimental/beacon_sim/belief_road_map_planner.hh"

#include <stack>

#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {

TEST(BeliefRoadMapPlannerTest, grid_road_map_no_backtrack) {
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
        .on_map_load_heading_uncertainty_rad = 0.1,
    };
    const auto &[road_map, ekf_slam, _] = create_grid_environment(ekf_config);
    const Eigen::Vector2d GOAL_STATE = {10, -5};
    constexpr BeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .num_start_connections = 1,
        .num_goal_connections = 1,
        .uncertainty_tolerance = std::nullopt,
        .max_num_edge_transforms = 1,
    };

    // Action
    const auto maybe_plan =
        compute_belief_road_map_plan(road_map, ekf_slam, {}, GOAL_STATE, OPTIONS);

    // Verification
    EXPECT_TRUE(maybe_plan.has_value());
    const auto &plan = maybe_plan.value();
    std::cout << "Num Nodes: " << plan.nodes.size() << std::endl;
    for (int i = 0; i < static_cast<int>(plan.nodes.size()); i++) {
        std::cout << i << " idx: " << plan.nodes.at(i) << std::endl;
    }
}

TEST(BeliefRoadMapPlannerTest, grid_road_map) {
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
        .on_map_load_heading_uncertainty_rad = 0.1,
    };
    const auto &[road_map, ekf_slam, _] = create_grid_environment(ekf_config);
    const Eigen::Vector2d GOAL_STATE = {10, -5};
    constexpr BeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .num_start_connections = 1,
        .num_goal_connections = 1,
        .uncertainty_tolerance = 1e-2,
        .max_num_edge_transforms = 1,
    };

    // Action
    const auto maybe_plan =
        compute_belief_road_map_plan(road_map, ekf_slam, {}, GOAL_STATE, OPTIONS);

    // Verification
    EXPECT_TRUE(maybe_plan.has_value());
    const auto &plan = maybe_plan.value();
    std::cout << "Num Nodes: " << plan.nodes.size() << std::endl;
    for (int i = 0; i < static_cast<int>(plan.nodes.size()); i++) {
        std::cout << i << " idx: " << plan.nodes.at(i) << std::endl;
    }
}

TEST(BeliefRoadMapPlannerTest, grid_road_map_with_unlikely_beacon) {
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
    constexpr double P_BEACON = 1e-7;
    const auto &[road_map, ekf_slam, potential] = create_grid_environment(ekf_config, P_BEACON);
    const Eigen::Vector2d GOAL_STATE = {10, -5};
    constexpr BeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .num_start_connections = 1,
        .num_goal_connections = 1,
        .uncertainty_tolerance = 1e-2,
        .max_num_edge_transforms = 1,
    };

    // Action
    const auto maybe_plan =
        compute_belief_road_map_plan(road_map, ekf_slam, potential, GOAL_STATE, OPTIONS);

    // Verification
    EXPECT_TRUE(maybe_plan.has_value());
    EXPECT_EQ(maybe_plan.value().nodes.size(), 6);
}

TEST(BeliefRoadMapPlannerTest, diamond_road_map_with_uncorrelated_beacons) {
    // Setup
    const EkfSlamConfig ekf_config{
        .max_num_beacons = 11,
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
    constexpr double P_LONE_BEACON = 0.5;
    constexpr double P_NO_STACKED_BEACON = 0.01;
    constexpr double P_STACKED_BEACON = 0.1;
    const auto &[road_map, ekf_slam, beacon_potential] = create_diamond_environment(
        ekf_config, P_LONE_BEACON, P_NO_STACKED_BEACON, P_STACKED_BEACON);
    const Eigen::Vector2d GOAL_STATE = {5, 7};
    constexpr BeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .num_start_connections = 1,
        .num_goal_connections = 1,
        .uncertainty_tolerance = std::nullopt,
        .max_num_edge_transforms = 10000,
    };

    // Action
    const auto maybe_plan =
        compute_belief_road_map_plan(road_map, ekf_slam, beacon_potential, GOAL_STATE, OPTIONS);

    // Verification
    EXPECT_TRUE(maybe_plan.has_value());
    const auto &plan = maybe_plan.value();
    for (const int node_id : plan.nodes) {
        EXPECT_NE(node_id, 1);
    }
}

TEST(BeliefRoadMapPlannerTest, diamond_road_map_with_correlated_beacons) {
    // Setup
    const EkfSlamConfig ekf_config{
        .max_num_beacons = 11,
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
    constexpr double P_LONE_BEACON = 0.5;
    constexpr double P_NO_STACKED_BEACON = 0.899;
    constexpr double P_STACKED_BEACON = 0.1;
    const auto &[road_map, ekf_slam, beacon_potential] = create_diamond_environment(
        ekf_config, P_LONE_BEACON, P_NO_STACKED_BEACON, P_STACKED_BEACON);
    const Eigen::Vector2d GOAL_STATE = {5, 7};
    constexpr BeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .num_start_connections = 1,
        .num_goal_connections = 1,
        .uncertainty_tolerance = std::nullopt,
        .max_num_edge_transforms = 1000,
    };

    // Action
    const auto maybe_plan =
        compute_belief_road_map_plan(road_map, ekf_slam, beacon_potential, GOAL_STATE, OPTIONS);

    // Verification
    EXPECT_TRUE(maybe_plan.has_value());
    const auto &plan = maybe_plan.value();
    for (const int node_id : plan.nodes) {
        EXPECT_NE(node_id, 2);
    }
}
TEST(ExpectedBeliefRoadMapPlannerTest, grid_road_map) {
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
        .on_map_load_heading_uncertainty_rad = 0.1,
    };

    constexpr double P_BEACON = 0.5;
    const auto &[road_map, ekf_slam, potential] = create_grid_environment(ekf_config, P_BEACON);
    const Eigen::Vector2d GOAL_STATE = {10, -5};
    constexpr ExpectedBeliefRoadMapOptions OPTIONS = {
        .max_path_length_ratio = 1.4,
        .start_goal_connection_radius_m = 6.0,
        .max_sensor_range_m = 3.0,
    };

    // Action
    const auto plan =
        compute_expected_belief_road_map_plan(road_map, ekf_slam, potential, GOAL_STATE, OPTIONS);

    // Verification
    const double pos_uncertainty_m_sq = ekf_config.on_map_load_position_uncertainty_m *
                                        ekf_config.on_map_load_position_uncertainty_m;
    const double heading_uncertainty_rad_sq = ekf_config.on_map_load_heading_uncertainty_rad *
                                              ekf_config.on_map_load_heading_uncertainty_rad;
    const double initial_cov_det =
        pos_uncertainty_m_sq * pos_uncertainty_m_sq * heading_uncertainty_rad_sq;

    EXPECT_LT(plan.expected_cov.determinant(), initial_cov_det);
}

}  // namespace robot::experimental::beacon_sim
