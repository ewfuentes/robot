
#include "experimental/beacon_sim/ekf_slam.hh"

#include <iostream>

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
namespace {
Eigen::VectorXd compute_gradient(const std::function<double(const Eigen::VectorXd &)> &f,
                                 const Eigen::VectorXd &eval_pt) {
    constexpr double PERTURB = 1e-3;
    auto perturb = [](const Eigen::VectorXd &pt, const int idx, const double step) {
        Eigen::VectorXd out = pt;
        out(idx) += step;
        return out;
    };
    Eigen::VectorXd out = Eigen::VectorXd::Zero(eval_pt.rows());
    for (int i = 0; i < eval_pt.rows(); i++) {
        const Eigen::VectorXd neg_perturb = perturb(eval_pt, i, -PERTURB);
        const Eigen::VectorXd pos_perturb = perturb(eval_pt, i, PERTURB);

        const double neg_eval = f(neg_perturb);
        const double pos_eval = f(pos_perturb);

        out(i) = (pos_eval - neg_eval) / (2 * PERTURB);
    }
    return out;
}
}  // namespace

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
        .bearing_measurement_noise_rad = 0.001,
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
        .bearing_measurement_noise_rad = 0.001,
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
        .initial_beacon_uncertainty_m = 1000,
        .along_track_process_noise_m_per_rt_meter = 0.1,
        .cross_track_process_noise_m_per_rt_meter = 0.01,
        .heading_process_noise_rad_per_rt_meter = 0.001,
        .beacon_pos_process_noise_m_per_rt_s = 0.1,
        .range_measurement_noise_m = 0.5,
        .bearing_measurement_noise_rad = 0.001,
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
    // After the initial update, it should be within 10 meters of the true position
    EXPECT_LT((maybe_beacon_in_local.value() - beacon_in_local).norm(), 10.0);
}

TEST(CreateMeasurementTest, incomplete_measurements_rejected) {
    // Setup
    const std::vector<BeaconObservation> observations{
        {.maybe_id = std::nullopt, .maybe_range_m = 10.0, .maybe_bearing_rad = 6.0},
        {.maybe_id = 123, .maybe_range_m = std::nullopt, .maybe_bearing_rad = 5.0},
        {.maybe_id = 456, .maybe_range_m = 20.0, .maybe_bearing_rad = std::nullopt},
        {.maybe_id = 789, .maybe_range_m = 30.0, .maybe_bearing_rad = 3.0}};

    const int ESTIMATE_DIM = 3 + 2 * observations.size();
    const EkfSlamEstimate estimate = {.mean = Eigen::VectorXd::Zero(ESTIMATE_DIM),
                                      .cov = Eigen::MatrixXd::Zero(ESTIMATE_DIM, ESTIMATE_DIM),
                                      .beacon_ids = {123, 456, 789}};

    // Action
    const auto [meas, pred, obs_mat] =
        detail::compute_measurement_and_prediction(observations, estimate);

    // Verification
    EXPECT_EQ(meas.rows(), 2);
    EXPECT_EQ(meas(0), observations.back().maybe_range_m.value());
    EXPECT_EQ(meas(1), observations.back().maybe_bearing_rad.value());
    EXPECT_EQ(obs_mat.rows(), 2);
    EXPECT_EQ(obs_mat.cols(), ESTIMATE_DIM);
}

TEST(CreateMeasurementTest, correct_observation_matrix) {
    // Setup
    const std::vector<BeaconObservation> observations{
        {.maybe_id = 123, .maybe_range_m = 10.0, .maybe_bearing_rad = 0.0},
        {.maybe_id = 456, .maybe_range_m = 20.0, .maybe_bearing_rad = std::numbers::pi / 2.0},
    };

    const int ESTIMATE_DIM = 3 + 2 * observations.size();
    // Place the robot at the origin with the +x axes aligned. Place a beacon at (10.0, 0.0) and
    // (0.0, 20.0) in the world frame.
    // This default constructs to the identity
    const Sophus::SE2d est_local_from_robot;
    const Eigen::Vector2d est_beacon_123_in_local{10.0, 0.0};
    const Eigen::Vector2d est_beacon_456_in_local{0.0, 20.0};

    const EkfSlamEstimate estimate = {
        .mean = (Eigen::VectorXd(ESTIMATE_DIM) << est_local_from_robot.log(),
                 est_beacon_123_in_local, est_beacon_456_in_local)
                    .finished(),
        .cov = Eigen::MatrixXd::Zero(ESTIMATE_DIM, ESTIMATE_DIM),
        .beacon_ids = {123, 456}};

    // Action
    const auto [meas, pred, obs_mat] =
        detail::compute_measurement_and_prediction(observations, estimate);

    // Verification
    EXPECT_EQ(meas.rows(), 4);
    EXPECT_EQ(pred.rows(), 4);
    EXPECT_EQ(obs_mat.rows(), 4);
    EXPECT_EQ(obs_mat.cols(), ESTIMATE_DIM);

    {
        // Check Beacon 123
        const Eigen::Vector2d est_beacon_in_robot =
            est_local_from_robot.inverse() * est_beacon_123_in_local;

        EXPECT_EQ(meas(0), observations.front().maybe_range_m.value());
        EXPECT_EQ(pred(0), est_beacon_in_robot.norm());

        EXPECT_EQ(meas(1), observations.front().maybe_bearing_rad.value());
        EXPECT_EQ(pred(1), std::atan2(est_beacon_in_robot.y(), est_beacon_in_robot.x()));
    }
    {
        // Check Beacon 456
        const Eigen::Vector2d est_beacon_in_robot =
            est_local_from_robot.inverse() * est_beacon_456_in_local;

        EXPECT_EQ(meas(2), observations.back().maybe_range_m.value());
        EXPECT_EQ(pred(2), est_beacon_in_robot.norm());

        EXPECT_EQ(meas(3), observations.back().maybe_bearing_rad.value());
        EXPECT_EQ(pred(3), std::atan2(est_beacon_in_robot.y(), est_beacon_in_robot.x()));
    }

    // Check the observation matrix using finite differences
    auto extract_prediction_entry = [&](const int i) {
        // Capture i by value to avoid a dangling reference
        auto compute_value = [&, i](const Eigen::VectorXd &pt) {
            EkfSlamEstimate perturb = estimate;
            perturb.mean = pt;
            const auto result = detail::compute_measurement_and_prediction(observations, perturb);
            return result.prediction(i);
        };
        return compute_value;
    };

    for (int i = 0; i < obs_mat.rows(); i++) {
        EXPECT_NEAR((obs_mat.row(i).transpose() -
                     compute_gradient(extract_prediction_entry(i), estimate.mean))
                        .norm(),
                    0.0, 1e-6);
    }
}
}  // namespace robot::experimental::beacon_sim
