
#include "experimental/beacon_sim/ekf_slam.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(EkfSlamTest, estimate_has_expected_dimensions) {
    // Setup
    constexpr auto CONFIG = EkfSlamConfig{
        .max_num_beacons = 11,
        .initial_beacon_uncertainty_m = 100,
        .along_track_process_noise_m_per_rt_meter = 0.1,
        .cross_track_process_noise_m_per_rt_meter = 0.01,
        .heading_process_noise_rad_per_rt_meter = 0.001,
        .beacon_pos_process_noise_m_per_rt_s = 0.1,
        .range_measurement_noise_m = 0.5,
        .bearing_measurment_noise_rad = 0.001,
    };
    constexpr int EXPECTED_ROBOT_DIM = 3;
    constexpr int EXPECTED_BEACON_DIM = 2;
    constexpr int EXPECTED_DIM = EXPECTED_ROBOT_DIM + EXPECTED_BEACON_DIM * CONFIG.max_num_beacons;

    // Action
    const EkfSlam ekf_slam(CONFIG);
    const EkfSlamEstimate &est = ekf_slam.estimate();

    // Verification
    EXPECT_EQ(est.mean.rows(), EXPECTED_DIM);
    EXPECT_EQ(est.cov.rows(), EXPECTED_DIM);
    EXPECT_EQ(est.cov.cols(), EXPECTED_DIM);

    // Expect that the robot states have no uncertainty
    EXPECT_EQ(est.robot_cov(), Eigen::Matrix3d::Zero());

    // Expect that no beacons are registered
    EXPECT_EQ(est.beacon_ids.size(), 0);
}

TEST(EkfSlamTest, prediction_updates_as_expected) {
    // Setup
    constexpr auto CONFIG = EkfSlamConfig{
        .max_num_beacons = 0,
        .initial_beacon_uncertainty_m = 100,
        .along_track_process_noise_m_per_rt_meter = 0.1,
        .cross_track_process_noise_m_per_rt_meter = 0.01,
        .heading_process_noise_rad_per_rt_meter = 0.001,
        .beacon_pos_process_noise_m_per_rt_s = 0.1,
        .range_measurement_noise_m = 0.5,
        .bearing_measurment_noise_rad = 0.001,
    };
    EkfSlam ekf_slam(CONFIG);
    const Sophus::SE2d old_robot_from_new_robot = Sophus::SE2d::exp({1.0, 2.0, 3.0});
    const EkfSlamEstimate initial_est = ekf_slam.estimate();

    // Action
    ekf_slam.predict(old_robot_from_new_robot);
    const EkfSlamEstimate &est = ekf_slam.estimate();

    // Verification
    const auto local_from_new_robot = initial_est.local_from_robot() * old_robot_from_new_robot;
    EXPECT_NEAR((local_from_new_robot.log() - est.local_from_robot().log()).norm(), 0.0, 1e-6);
}

TEST(EkfSlamTest, measurement_updates_as_expected) {
    // Setup
    constexpr auto CONFIG = EkfSlamConfig{
        .max_num_beacons = 1,
        .initial_beacon_uncertainty_m = 100,
        .along_track_process_noise_m_per_rt_meter = 0.1,
        .cross_track_process_noise_m_per_rt_meter = 0.01,
        .heading_process_noise_rad_per_rt_meter = 0.001,
        .beacon_pos_process_noise_m_per_rt_s = 0.1,
        .range_measurement_noise_m = 0.5,
        .bearing_measurment_noise_rad = 0.001,
    };

    constexpr int BEACON_ID = 10;
    const Eigen::Vector2d beacon_in_local{3, 4};
    EkfSlam ekf_slam(CONFIG);
    const auto initial_est = ekf_slam.predict(Sophus::SE2d());

    const auto beacon_in_robot = initial_est.local_from_robot().inverse() * beacon_in_local;
    const BeaconObservation obs = {
        .maybe_id = BEACON_ID,
        .maybe_range_m = beacon_in_robot.norm(),
        .maybe_bearing_rad = std::atan2(beacon_in_robot.y(), beacon_in_robot.x())};

    // Action
    const auto est = ekf_slam.update({obs});

    // Verification
    const auto maybe_beacon_in_local = est.beacon_in_local(BEACON_ID);
    EXPECT_TRUE(maybe_beacon_in_local.has_value());
    EXPECT_NEAR((maybe_beacon_in_local.value() - beacon_in_local).norm(), 0.0, 1e-6);
}
}  // namespace robot::experimental::beacon_sim
