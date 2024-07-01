
#include "experimental/beacon_sim/make_belief_updater.hh"

#include "experimental/beacon_sim/test_helpers.hh"
#include "gmock/gmock.h"
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

    // Action
    const auto belief_updater = make_landmark_belief_updater(
        road_map, MAX_SENSOR_RANGE_M, std::nullopt, ekf_slam, beacon_potential, TypeParam::value);
    const LandmarkRobotBelief initial_belief = {
        .local_from_robot = local_from_robot,
        .log_probability_mass_tracked = 0,
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

TYPED_TEST(MakeBeliefUpdaterTest, make_subsampled_landmark_belief_updater_test) {
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
    constexpr SampledLandmarkBeliefOptions sampled_belief_options = {
        .max_num_components = 1,
        .seed = 0,
    };

    // Action
    const auto belief_updater =
        make_landmark_belief_updater(road_map, MAX_SENSOR_RANGE_M, sampled_belief_options, ekf_slam,
                                     beacon_potential, TypeParam::value);
    const LandmarkRobotBelief initial_belief = {
        .local_from_robot = local_from_robot,
        .log_probability_mass_tracked = 0,
        .belief_from_config = {{"?",
                                {.cov_in_robot = ekf_slam.estimate().robot_cov(),
                                 .log_config_prob = 0}}},
    };
    const auto updated_belief = belief_updater(initial_belief, START_NODE_IDX, END_NODE_IDX);

    // Verification
    EXPECT_EQ(updated_belief.belief_from_config.size(), 1);
    EXPECT_TRUE(updated_belief.belief_from_config.contains("0") ||
                updated_belief.belief_from_config.contains("1"));
    if (updated_belief.belief_from_config.contains("0")) {
        EXPECT_NEAR(updated_belief.belief_from_config.at("0").log_config_prob, 0.0, 1e-6);
        EXPECT_NEAR(updated_belief.log_probability_mass_tracked, std::log(1 - P_BEACON), 1e-6);
    } else if (updated_belief.belief_from_config.contains("1")) {
        EXPECT_NEAR(updated_belief.belief_from_config.at("1").log_config_prob, 0.0, 1e-6);
        EXPECT_NEAR(updated_belief.log_probability_mass_tracked, std::log(P_BEACON), 1e-6);
    }
}

TEST(TransformComputerTest, check_key_generation) {
    // Setup
    const EkfSlamConfig ekf_config{
        .max_num_beacons = 15,
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
    constexpr double P_LONE_BEACON = 0.2;
    constexpr double P_NO_STACKED_BEACON = 0.2;
    constexpr double P_STACKED_BEACON = 0.2;
    constexpr double MAX_SENSOR_RANGE_M = 3;
    const auto &[road_map, ekf_slam, beacon_potential] = create_diamond_environment(
        ekf_config, P_LONE_BEACON, P_NO_STACKED_BEACON, P_STACKED_BEACON);
    const liegroups::SE2 local_from_initial_robot(0.0, road_map.point(0));
    const Eigen::Vector2d goal_in_local(road_map.point(2));

    const TransformComputer computer(local_from_initial_robot, goal_in_local, ekf_config,
                                     ekf_slam.estimate(), MAX_SENSOR_RANGE_M,
                                     TransformType::COVARIANCE, {11, 12, 13, 14, 15}, {});

    // Action + Verification
    EXPECT_EQ(computer.key(0, beacon_potential.members()), "?00000?????");
    EXPECT_EQ(computer.key(1, beacon_potential.members()), "?10000?????");
    EXPECT_EQ(computer.key(2, beacon_potential.members()), "?01000?????");
    EXPECT_EQ(computer.key(31, beacon_potential.members()), "?11111?????");
}

TEST(TransformComputerTest, check_consistent_configs) {
    // Setup
    const EkfSlamConfig ekf_config{
        .max_num_beacons = 15,
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
    constexpr double P_LONE_BEACON = 0.2;
    constexpr double P_NO_STACKED_BEACON = 0.2;
    constexpr double P_STACKED_BEACON = 0.2;
    constexpr double MAX_SENSOR_RANGE_M = 3;
    const auto &[road_map, ekf_slam, beacon_potential] = create_diamond_environment(
        ekf_config, P_LONE_BEACON, P_NO_STACKED_BEACON, P_STACKED_BEACON);
    const liegroups::SE2 local_from_initial_robot(0.0, road_map.point(0));
    const Eigen::Vector2d goal_in_local(road_map.point(2));

    const TransformComputer computer(local_from_initial_robot, goal_in_local, ekf_config,
                                     ekf_slam.estimate(), MAX_SENSOR_RANGE_M,
                                     TransformType::COVARIANCE, {11, 12, 13, 14, 15}, {});

    // Action
    const auto configs = computer.consistent_configs("011???01101", beacon_potential.members());

    // Verification
    EXPECT_EQ(configs.size(), 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_THAT(configs, ::testing::Contains((i << 2) + 3));
    }
}

}  // namespace robot::experimental::beacon_sim
