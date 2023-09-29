
#include "experimental/beacon_sim/make_belief_updater.hh"

#include "common/math/redheffer_star.hh"
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
        MAX_NUM_TRANSFORMS, TransformType::COVARIANCE);

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
        MAX_NUM_TRANSFORMS, TransformType::COVARIANCE);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

TEST(MakeBeliefUpdaterTest, compute_edge_transform_no_measurements_information) {
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
        MAX_NUM_TRANSFORMS, TransformType::INFORMATION);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

TEST(MakeBeliefUpdaterTest, compute_edge_transform_with_measurement_information) {
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
        MAX_NUM_TRANSFORMS, TransformType::INFORMATION);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

TEST(MakeBeliefUpdaterTest, stack_covariance_updates_with_observation) {
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

    const auto &[road_map, ekf_slam, _] = create_grid_environment(ekf_config);
    constexpr double MAX_SENSOR_RANGE_M = 3.0;

    // Traverse the (3, 0) edge that contains a beacon
    constexpr int START_IDX = 3;
    constexpr int END_IDX = 0;
    const liegroups::SE2 local_from_start(0.0, road_map.points.at(START_IDX));

    const auto &[local_from_end, transform] = compute_edge_belief_transform(
        local_from_start, road_map.points.at(END_IDX), ekf_slam.config(), ekf_slam.estimate(),
        {{GRID_BEACON_ID}}, MAX_SENSOR_RANGE_M, TransformType::COVARIANCE);
    (void)local_from_end;

    // Action
    const TypedTransformVector stacked_transforms = [&transform = transform]() {
        TypedTransformVector stacked = {transform};
        for (int i = 0; i < 30; i++) {
            stacked.push_back((stacked.back() * stacked.back()).value());
        }
        return stacked;
    }();

    // Verification

    std::cout << stacked_transforms.size() << std::endl;

    for (int i = 0; i < static_cast<int>(stacked_transforms.size()); i++) {
        std::cout << "===============" << i << " " << (1 << i) << " transforms" << std::endl;
        std::cout << std::get<ScatteringTransform<TransformType::COVARIANCE>>(
                         stacked_transforms.at(i))
                  << std::endl;

        Eigen::JacobiSVD<ScatteringTransformBase> svd(
            std::get<ScatteringTransform<TransformType::COVARIANCE>>(stacked_transforms.at(i)));
        std::cout << "SVD Success? " << (svd.info() == Eigen::ComputationInfo::Success)
                  << std::endl;
        std::cout << "SV: " << svd.singularValues().transpose() << std::endl;
    }
}

TEST(MakeBeliefUpdaterTest, stack_covariance_updates_with_no_observation) {
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

    const auto &[road_map, ekf_slam, _] = create_grid_environment(ekf_config);
    constexpr double MAX_SENSOR_RANGE_M = 3.0;

    // Traverse the (3, 0) edge that contains a beacon
    constexpr int START_IDX = 3;
    constexpr int END_IDX = 0;
    const liegroups::SE2 local_from_start(0.0, road_map.points.at(START_IDX));

    const auto &[local_from_end, transform] = compute_edge_belief_transform(
        local_from_start, road_map.points.at(END_IDX), ekf_slam.config(), ekf_slam.estimate(), {{}},
        MAX_SENSOR_RANGE_M, TransformType::INFORMATION);
    (void)local_from_end;

    // Action
    const TypedTransformVector stacked_transforms = [&transform = transform]() {
        TypedTransformVector stacked = {transform};
        for (int i = 0; i < 30; i++) {
            stacked.push_back((stacked.back() * stacked.back()).value());
        }
        return stacked;
    }();

    // Verification

    std::cout << stacked_transforms.size() << std::endl;

    for (int i = 0; i < static_cast<int>(stacked_transforms.size()); i++) {
        std::cout << "===============" << i << " " << (1 << i) << " transforms" << std::endl;
        std::cout << std::get<ScatteringTransform<TransformType::INFORMATION>>(
                         stacked_transforms.at(i))
                  << std::endl;

        Eigen::JacobiSVD<ScatteringTransformBase> svd(
            std::get<ScatteringTransform<TransformType::INFORMATION>>(stacked_transforms.at(i)));
        std::cout << "SVD Success? " << (svd.info() == Eigen::ComputationInfo::Success)
                  << std::endl;
        std::cout << "SV: " << svd.singularValues().transpose() << std::endl;
    }
}
}  // namespace robot::experimental::beacon_sim
