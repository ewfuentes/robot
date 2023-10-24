
#include "experimental/beacon_sim/make_belief_updater.hh"

#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

template <TransformType T>
struct EnumValue {
    static const TransformType value = T;
};

template <typename T>
class MakeBeliefUpdaterTest : public ::testing::Test {
    static const TransformType value = T::value;
};

using MyTypes =
    ::testing::Types<EnumValue<TransformType::INFORMATION>, EnumValue<TransformType::COVARIANCE>>;
TYPED_TEST_SUITE(MakeBeliefUpdaterTest, MyTypes);

TYPED_TEST(MakeBeliefUpdaterTest, compute_edge_transform_no_measurements) {
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
    const liegroups::SE2 local_from_robot = liegroups::SE2::trans(road_map.point(START_NODE_IDX));
    const Eigen::Vector2d end_pos = road_map.point(END_NODE_IDX);

    // Action
    const auto edge_belief_transform = compute_edge_belief_transform(
        local_from_robot, end_pos, ekf_slam.config(), ekf_slam.estimate(), {}, MAX_SENSOR_RANGE_M,
        MAX_NUM_TRANSFORMS, TypeParam::value);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

TYPED_TEST(MakeBeliefUpdaterTest, compute_edge_transform_with_measurement) {
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
    const liegroups::SE2 local_from_robot = liegroups::SE2::trans(road_map.point(START_NODE_IDX));
    const Eigen::Vector2d end_pos = road_map.point(END_NODE_IDX);

    // Action
    const auto edge_belief_transform = compute_edge_belief_transform(
        local_from_robot, end_pos, ekf_slam.config(), ekf_slam.estimate(), {}, MAX_SENSOR_RANGE_M,
        MAX_NUM_TRANSFORMS, TypeParam::value);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

TYPED_TEST(MakeBeliefUpdaterTest, compute_edge_transform_no_measurements_information) {
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
    const liegroups::SE2 local_from_robot = liegroups::SE2::trans(road_map.point(START_NODE_IDX));
    const Eigen::Vector2d end_pos = road_map.point(END_NODE_IDX);

    // Action
    const auto edge_belief_transform = compute_edge_belief_transform(
        local_from_robot, end_pos, ekf_slam.config(), ekf_slam.estimate(), {}, MAX_SENSOR_RANGE_M,
        MAX_NUM_TRANSFORMS, TypeParam::value);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

TYPED_TEST(MakeBeliefUpdaterTest, compute_edge_transform_with_measurement_information) {
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
    const liegroups::SE2 local_from_robot = liegroups::SE2::trans(road_map.point(START_NODE_IDX));
    const Eigen::Vector2d end_pos = road_map.point(END_NODE_IDX);

    // Action
    const auto edge_belief_transform = compute_edge_belief_transform(
        local_from_robot, end_pos, ekf_slam.config(), ekf_slam.estimate(), {}, MAX_SENSOR_RANGE_M,
        MAX_NUM_TRANSFORMS, TypeParam::value);

    // Verification
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().x(), end_pos.x());
    EXPECT_EQ(edge_belief_transform.local_from_robot.translation().y(), end_pos.y());
}

TYPED_TEST(MakeBeliefUpdaterTest, make_landmark_belief_updater_test) {
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
    constexpr double P_BEACON = 0.75;
    const auto &[road_map, ekf_slam, beacon_potential] =
        create_grid_environment(ekf_config, P_BEACON);

    constexpr int START_NODE_IDX = 3;
    constexpr int END_NODE_IDX = 0;
    const liegroups::SE2 local_from_robot = liegroups::SE2::trans(road_map.point(START_NODE_IDX));
    const Eigen::Vector2d end_pos = road_map.point(END_NODE_IDX);

    // Action
    const auto belief_updater = make_landmark_belief_updater(road_map, MAX_SENSOR_RANGE_M, ekf_slam,
                                                             beacon_potential, TypeParam::value);
    const LandmarkRobotBelief initial_belief = {
        .local_from_robot = local_from_robot,
        .belief_from_config = {{"?",
                                {.cov_in_robot = ekf_slam.estimate().robot_cov(),
                                 .log_config_prob = 0}}},
    };
    const auto updated_belief = belief_updater(initial_belief, START_NODE_IDX, END_NODE_IDX);

    // Verification
    EXPECT_TRUE(updated_belief.belief_from_config.contains("0"));
    EXPECT_TRUE(updated_belief.belief_from_config.contains("1"));

    EXPECT_NEAR(updated_belief.belief_from_config.at("0").log_config_prob, std::log(1 - P_BEACON),
                1e-6);
    EXPECT_NEAR(updated_belief.belief_from_config.at("1").log_config_prob, std::log(P_BEACON),
                1e-6);
    EXPECT_GT(updated_belief.belief_from_config.at("0").cov_in_robot.determinant(),
              updated_belief.belief_from_config.at("1").cov_in_robot.determinant());
}

}  // namespace robot::experimental::beacon_sim
