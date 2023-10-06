#include "experimental/beacon_sim/make_eigen_bounds.hh"

#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(MakeEigenBoundsTest, step_eigen_bound_no_information) {
    // Setup 
    constexpr double start_info_eigen_value = 1.0;
    constexpr double ub_dynamics_eigen_value = 1.0;
    constexpr double ub_measurement_eigen_value = 0.0;
    constexpr double lower_process_noise_eigen_value = 1.0;

    // Action
    double newbound = step_lower_eigen_bound(start_info_eigen_value, ub_dynamics_eigen_value, ub_measurement_eigen_value, lower_process_noise_eigen_value);

    // Verification
    // lower bound should increase as more information is required travel the edge (as no info is gained and process noise is added)
    ASSERT_GT(newbound, start_info_eigen_value);
}
TEST(MakeEigenBoundsTest, step_eigen_bound_information) {
    // Setup 
    constexpr double start_info_eigen_value = 1.0;
    constexpr double ub_dynamics_eigen_value = 1.0;
    constexpr double ub_measurement_eigen_value = 1.0;
    constexpr double lower_process_noise_eigen_value = 1.0;

    // Action
    double newbound = step_lower_eigen_bound(start_info_eigen_value, ub_dynamics_eigen_value, ub_measurement_eigen_value, lower_process_noise_eigen_value);

    // Verification
    // lower bound should decrease as information is added along the edge that outweighs the process noise 
    ASSERT_LT(newbound, start_info_eigen_value);
}

TEST(MakeEigenBoundsTest, compute_edge_transform_no_measurements) {
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
    const auto &[road_map, ekf_slam, _] = create_grid_environment(ekf_config);

    constexpr int START_NODE_IDX = 6;
    constexpr int END_NODE_IDX = 3;
    const liegroups::SE2 local_from_robot =
        liegroups::SE2::trans(road_map.points.at(END_NODE_IDX));

    std::cout << "End node position is " << local_from_robot.translation() << std::endl;
    const Eigen::Vector2d start_pos = road_map.points.at(START_NODE_IDX);
    std::cout << "Start node position is " << start_pos << std::endl;

    // Action
    const auto edge_belief_transform = compute_backwards_edge_belief_transform(
        local_from_robot, start_pos, ekf_slam.config(), ekf_slam.estimate(), {}, MAX_SENSOR_RANGE_M,
        TransformType::INFORMATION);

    // Verification
    EXPECT_EQ(edge_belief_transform, 0.0);
}


}  // namespace robot::experimental::beacon_sim
