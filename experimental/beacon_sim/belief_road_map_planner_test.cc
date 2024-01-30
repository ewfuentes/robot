
#include "experimental/beacon_sim/belief_road_map_planner.hh"

#include <optional>
#include <stack>

#include "common/check.hh"
#include "common/math/logsumexp.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"
#include "planning/probabilistic_road_map.hh"
#include "planning/road_map.hh"

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
    constexpr BeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .uncertainty_tolerance = std::nullopt,
        .max_num_edge_transforms = 1,
        .timeout = std::nullopt,
    };

    // Action
    const auto maybe_plan = compute_belief_road_map_plan(road_map, ekf_slam, {}, OPTIONS);

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
    constexpr BeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .uncertainty_tolerance = 1e-2,
        .max_num_edge_transforms = 1,
        .timeout = std::nullopt,
    };

    // Action
    const auto maybe_plan = compute_belief_road_map_plan(road_map, ekf_slam, {}, OPTIONS);

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
    constexpr BeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .uncertainty_tolerance = 1e-2,
        .max_num_edge_transforms = 1,
        .timeout = std::nullopt,
    };

    // Action
    const auto maybe_plan = compute_belief_road_map_plan(road_map, ekf_slam, potential, OPTIONS);

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
    constexpr BeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .uncertainty_tolerance = std::nullopt,
        .max_num_edge_transforms = 10000,
        .timeout = std::nullopt,
    };

    // Action
    const auto maybe_plan =
        compute_belief_road_map_plan(road_map, ekf_slam, beacon_potential, OPTIONS);

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
    constexpr BeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .uncertainty_tolerance = std::nullopt,
        .max_num_edge_transforms = 1000,
        .timeout = std::nullopt,
    };

    // Action
    const auto maybe_plan =
        compute_belief_road_map_plan(road_map, ekf_slam, beacon_potential, OPTIONS);

    // Verification
    EXPECT_TRUE(maybe_plan.has_value());
    const auto &plan = maybe_plan.value();
    for (const int node_id : plan.nodes) {
        EXPECT_NE(node_id, 2);
    }
}
TEST(PathConstrainedBeliefRoadMapPlannerTest, grid_road_map) {
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
    constexpr PathConstrainedBeliefRoadMapOptions OPTIONS = {
        .max_path_length_ratio = 1.4,
        .max_sensor_range_m = 3.0,
    };

    // Action
    const auto plan =
        compute_path_constrained_belief_road_map_plan(road_map, ekf_slam, potential, OPTIONS);

    // Verification
    const double pos_uncertainty_m_sq = ekf_config.on_map_load_position_uncertainty_m *
                                        ekf_config.on_map_load_position_uncertainty_m;
    const double heading_uncertainty_rad_sq = ekf_config.on_map_load_heading_uncertainty_rad *
                                              ekf_config.on_map_load_heading_uncertainty_rad;
    const double initial_cov_det =
        pos_uncertainty_m_sq * pos_uncertainty_m_sq * heading_uncertainty_rad_sq;

    EXPECT_LT(plan.expected_cov.determinant(), initial_cov_det);
}

TEST(LandmarkBeliefRoadMapPlannerTest, grid_road_map_low_prob_beacon) {
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
    constexpr LandmarkBeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .uncertainty_size_options =
            LandmarkBeliefRoadMapOptions::ValueAtRiskDeterminant{.percentile = 0.95},
        .sampled_belief_options = std::nullopt,
        .timeout = std::nullopt,
    };
    const std::vector<std::optional<int>> expected_path = {
        planning::RoadMap::START_IDX, 7, std::nullopt, std::nullopt, 2,
        planning::RoadMap::GOAL_IDX};

    // Action
    const auto plan = compute_landmark_belief_road_map_plan(road_map, ekf_slam, potential, OPTIONS);

    // Verification
    EXPECT_TRUE(plan.has_value());
    EXPECT_EQ(plan->nodes.size(), expected_path.size());
    for (int i = 0; i < static_cast<int>(plan->nodes.size()); i++) {
        if (expected_path.at(i).has_value()) {
            EXPECT_EQ(plan.value().nodes.at(i), expected_path.at(i).value());
        }
    }
}

TEST(LandmarkBeliefRoadMapPlannerTest, grid_road_map_high_prob_beacon) {
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

    constexpr double P_BEACON = 0.99;
    const auto &[road_map, ekf_slam, potential] = create_grid_environment(ekf_config, P_BEACON);
    constexpr LandmarkBeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .uncertainty_size_options =
            LandmarkBeliefRoadMapOptions::ValueAtRiskDeterminant{.percentile = 0.95},
        .sampled_belief_options = std::nullopt,
        .timeout = std::nullopt,
    };
    const std::vector<std::optional<int>> expected_path = {
        planning::RoadMap::START_IDX, 7, std::nullopt, 3, 0, 1, 2, planning::RoadMap::GOAL_IDX};

    // Action
    const auto plan = compute_landmark_belief_road_map_plan(road_map, ekf_slam, potential, OPTIONS);

    // Verification
    EXPECT_TRUE(plan.has_value());
    EXPECT_EQ(plan->nodes.size(), expected_path.size());
    for (int i = 0; i < static_cast<int>(plan->nodes.size()); i++) {
        if (expected_path.at(i).has_value()) {
            EXPECT_EQ(plan.value().nodes.at(i), expected_path.at(i).value());
        }
    }
}

TEST(LandmarkBeliefRoadMapPlannerTest, diamond_road_map_independent_beacons) {
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
        .on_map_load_heading_uncertainty_rad = 0.1,
    };

    constexpr double P_LONE_BEACON = 0.5;
    constexpr double P_STACKED_BEACON = 0.1;
    const double P_NO_STACK_BEACON = std::pow(P_STACKED_BEACON, 10);
    const auto &[road_map, ekf_slam, potential] =
        create_diamond_environment(ekf_config, P_LONE_BEACON, P_NO_STACK_BEACON, P_STACKED_BEACON);
    constexpr LandmarkBeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .uncertainty_size_options =
            LandmarkBeliefRoadMapOptions::ValueAtRiskDeterminant{.percentile = 0.95},
        .sampled_belief_options = std::nullopt,
        .timeout = std::nullopt,
    };
    const std::vector<std::optional<int>> expected_path = {planning::RoadMap::START_IDX, 0, 2, 3,
                                                           planning::RoadMap::GOAL_IDX};

    // Action
    const auto plan = compute_landmark_belief_road_map_plan(road_map, ekf_slam, potential, OPTIONS);

    // Verification
    EXPECT_TRUE(plan.has_value());
    EXPECT_EQ(plan->nodes.size(), expected_path.size());
    for (int i = 0; i < static_cast<int>(plan->nodes.size()); i++) {
        if (expected_path.at(i).has_value()) {
            EXPECT_EQ(plan.value().nodes.at(i), expected_path.at(i).value());
        }
    }
}

TEST(LandmarkBeliefRoadMapPlannerTest, diamond_road_map_correlated_beacons) {
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
        .on_map_load_heading_uncertainty_rad = 0.1,
    };

    constexpr double P_LONE_BEACON = 0.5;
    constexpr double P_STACKED_BEACON = 0.1;
    const double P_NO_STACK_BEACON = 0.6;
    const auto &[road_map, ekf_slam, potential] =
        create_diamond_environment(ekf_config, P_LONE_BEACON, P_NO_STACK_BEACON, P_STACKED_BEACON);
    constexpr LandmarkBeliefRoadMapOptions OPTIONS = {
        .max_sensor_range_m = 3.0,
        .uncertainty_size_options =
            LandmarkBeliefRoadMapOptions::ValueAtRiskDeterminant{
                .percentile = 0.95,
            },
        .sampled_belief_options = std::nullopt,
        .timeout = std::nullopt,
    };
    const std::vector<std::optional<int>> expected_path = {planning::RoadMap::START_IDX, 0, 1, 3,
                                                           planning::RoadMap::GOAL_IDX};

    // Action
    const auto plan = compute_landmark_belief_road_map_plan(road_map, ekf_slam, potential, OPTIONS);

    // Verification
    EXPECT_TRUE(plan.has_value());
    EXPECT_EQ(plan->nodes.size(), expected_path.size());
    for (int i = 0; i < static_cast<int>(plan->nodes.size()); i++) {
        if (expected_path.at(i).has_value()) {
            EXPECT_EQ(plan.value().nodes.at(i), expected_path.at(i).value());
        }
    }
}

TEST(ExpectedBeliefRoadMapPlannerTest, diamond_road_map_correlated_beacons) {
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
        .on_map_load_heading_uncertainty_rad = 0.1,
    };

    constexpr double P_LONE_BEACON = 0.5;
    constexpr double P_STACKED_BEACON = 0.1;
    const double P_NO_STACK_BEACON = 0.6;
    const auto &[road_map, ekf_slam, potential] =
        create_diamond_environment(ekf_config, P_LONE_BEACON, P_NO_STACK_BEACON, P_STACKED_BEACON);
    constexpr ExpectedBeliefRoadMapOptions OPTIONS = {
        .num_configuration_samples = 100,
        .seed = 1234,
        .brm_options = {
            .max_sensor_range_m = 3.0,
            .uncertainty_tolerance = std::nullopt,
            .max_num_edge_transforms = std::numeric_limits<int>::max(),
            .timeout = std::nullopt,
        }};
    const std::vector<std::optional<int>> expected_path = {planning::RoadMap::START_IDX, 0, 1, 3,
                                                           planning::RoadMap::GOAL_IDX};

    // Action
    const auto plan = compute_expected_belief_road_map_plan(road_map, ekf_slam, potential, OPTIONS);

    const auto marginals = potential.log_marginals(potential.members());

    // Verification
    EXPECT_TRUE(plan.has_value());
    EXPECT_EQ(plan->nodes.size(), expected_path.size());
    for (int i = 0; i < static_cast<int>(plan->nodes.size()); i++) {
        if (expected_path.at(i).has_value()) {
            EXPECT_EQ(plan.value().nodes.at(i), expected_path.at(i).value());
        }
    }
}

}  // namespace robot::experimental::beacon_sim
