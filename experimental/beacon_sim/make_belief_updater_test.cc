
#include "experimental/beacon_sim/make_belief_updater.hh"

#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(MakeBeliefUpdaterTest, compute_edge_transform_no_measurements) {
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
    constexpr double MAX_SENSOR_RANGE_M = 3.0;
    constexpr int MAX_NUM_TRANSFORMS = 10;
    const auto &[road_map, ekf_slam, _] = create_grid_environment(ekf_config);

    constexpr int START_NODE_IDX = 6;
    constexpr int END_NODE_IDX = 3;
    const liegroups::SE2 local_from_robot =
        liegroups::SE2::trans(road_map.points.at(START_NODE_IDX));
    const Eigen::Vector2d end_pos = road_map.points.at(END_NODE_IDX);

    // Action
    const auto edge_belief_transform = compute_edge_belief_transform(
        local_from_robot, end_pos, ekf_slam.config(), ekf_slam.estimate(), {}, MAX_SENSOR_RANGE_M,
        MAX_NUM_TRANSFORMS);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

TEST(MakeBeliefUpdaterTest, compute_edge_transform_with_measurement) {
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
    constexpr double MAX_SENSOR_RANGE_M = 3.0;
    constexpr int MAX_NUM_TRANSFORMS = 10;
    const auto &[road_map, ekf_slam, _] = create_grid_environment(ekf_config);

    constexpr int START_NODE_IDX = 3;
    constexpr int END_NODE_IDX = 0;
    const liegroups::SE2 local_from_robot =
        liegroups::SE2::trans(road_map.points.at(START_NODE_IDX));
    const Eigen::Vector2d end_pos = road_map.points.at(END_NODE_IDX);

    // Action
    const auto edge_belief_transform = compute_edge_belief_transform(
        local_from_robot, end_pos, ekf_slam.config(), ekf_slam.estimate(), {}, MAX_SENSOR_RANGE_M,
        MAX_NUM_TRANSFORMS);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

}  // namespace robot::experimental::beacon_sim
