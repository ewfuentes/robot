
#include "experimental/beacon_sim/ekf_slam.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
constexpr double TOL = 1e-6;
using namespace std::literals::chrono_literals;

class EkfSlamTestHelper {
   public:
    static EkfSlamEstimate &estimate(InOut<EkfSlam> ekf_slam) { return ekf_slam->estimate_; }
};

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
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 0.001,
        .beacon_pos_process_noise_m_per_rt_s = 0.1,
        .range_measurement_noise_m = 0.5,
        .bearing_measurement_noise_rad = 0.001,
    };
    constexpr int EXPECTED_ROBOT_DIM = 3;
    constexpr int EXPECTED_BEACON_DIM = 2;
    constexpr int EXPECTED_DIM = EXPECTED_ROBOT_DIM + EXPECTED_BEACON_DIM * CONFIG.max_num_beacons;

    // Action
    const EkfSlam ekf_slam(CONFIG, time::RobotTimestamp());
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
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 0.001,
        .beacon_pos_process_noise_m_per_rt_s = 0.1,
        .range_measurement_noise_m = 0.5,
        .bearing_measurement_noise_rad = 0.001,
    };
    EkfSlam ekf_slam(CONFIG, time::RobotTimestamp());
    const liegroups::SE2 old_robot_from_new_robot = liegroups::SE2::exp({1.0, 2.0, 3.0});
    const EkfSlamEstimate initial_est = ekf_slam.estimate();

    // Action
    ekf_slam.predict(1s + time::RobotTimestamp(), old_robot_from_new_robot);
    const EkfSlamEstimate &est = ekf_slam.estimate();

    // Verification
    const auto local_from_new_robot = initial_est.local_from_robot() * old_robot_from_new_robot;
    EXPECT_NEAR((local_from_new_robot.log() - est.local_from_robot().log()).norm(), 0.0, TOL);
}

TEST(EkfSlamTest, measurement_updates_as_expected) {
    // Setup
    constexpr auto CONFIG = EkfSlamConfig{
        .max_num_beacons = 1,
        .initial_beacon_uncertainty_m = 1000,
        .along_track_process_noise_m_per_rt_meter = 0.1,
        .cross_track_process_noise_m_per_rt_meter = 0.01,
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 0.001,
        .beacon_pos_process_noise_m_per_rt_s = 0.1,
        .range_measurement_noise_m = 0.5,
        .bearing_measurement_noise_rad = 0.001,
    };

    constexpr int BEACON_ID = 10;
    const Eigen::Vector2d beacon_in_local{3, 4};
    EkfSlam ekf_slam(CONFIG, time::RobotTimestamp());
    const auto initial_est = ekf_slam.predict(time::RobotTimestamp(), liegroups::SE2());

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

TEST(EkfSlamTest, identity_update_does_not_change_estimate) {
    // Setup
    constexpr auto CONFIG = EkfSlamConfig{
        .max_num_beacons = 1,
        .initial_beacon_uncertainty_m = 1e3,
        .along_track_process_noise_m_per_rt_meter = 0.1,
        .cross_track_process_noise_m_per_rt_meter = 0.01,
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 0.001,
        .beacon_pos_process_noise_m_per_rt_s = 1e-6,
        .range_measurement_noise_m = 1e-3,
        .bearing_measurement_noise_rad = 1e-6,
    };

    EkfSlam ekf_slam(CONFIG, time::RobotTimestamp());

    // Action
    constexpr int NUM_ITERS = 1000;
    constexpr time::RobotTimestamp::duration DT = 100ms;
    for (int i = 0; i < NUM_ITERS; i++) {
        ekf_slam.predict(i * DT + time::RobotTimestamp(), liegroups::SE2());
    }

    const auto est = ekf_slam.estimate();

    // Verification
    EXPECT_NEAR((est.local_from_robot().log() - Eigen::Vector3d::Zero()).norm(), 0.0, 1e-6);
    // We should probably accumulate some uncertainty in position/rotation as a function rotation
    EXPECT_NEAR(est.robot_cov()(0, 0), 0.0, TOL);
    EXPECT_NEAR(est.robot_cov()(1, 1), 0.0, TOL);
    EXPECT_NEAR(est.robot_cov()(2, 2), 0.0, TOL);
}

TEST(EkfSlamTest, position_estimate_converges_given_known_beacon) {
    // Setup
    constexpr auto CONFIG = EkfSlamConfig{
        .max_num_beacons = 2,
        .initial_beacon_uncertainty_m = 1e-6,
        .along_track_process_noise_m_per_rt_meter = 1.0,
        .cross_track_process_noise_m_per_rt_meter = 1.0,
        .pos_process_noise_m_per_rt_s = 0.1,
        .heading_process_noise_rad_per_rt_meter = 1.0,
        .heading_process_noise_rad_per_rt_s = 0.01,
        .beacon_pos_process_noise_m_per_rt_s = 1e-9,
        .range_measurement_noise_m = 1e-3,
        .bearing_measurement_noise_rad = 1e-3,
    };

    constexpr std::array<int, 2> BEACON_IDS = {10, 11};
    const std::array<Eigen::Vector2d, 2> beacons_in_local{{{10.0, 20.0}, {10.0, 30.0}}};
    const liegroups::SE2 expected_local_from_robot =
        liegroups::SE2(std::numbers::pi / 3.0, {5.0, 12.0});

    EkfSlam ekf_slam(CONFIG, time::RobotTimestamp());

    EkfSlamTestHelper::estimate(make_in_out(ekf_slam)).mean(Eigen::seqN(3, 2)) =
        beacons_in_local[0];
    EkfSlamTestHelper::estimate(make_in_out(ekf_slam)).mean(Eigen::seqN(5, 2)) =
        beacons_in_local[1];
    EkfSlamTestHelper::estimate(make_in_out(ekf_slam)).cov.topLeftCorner(2, 2) =
        1e2 * Eigen::Matrix2d::Identity();
    EkfSlamTestHelper::estimate(make_in_out(ekf_slam)).cov(2, 2) = 3.0;
    std::copy(BEACON_IDS.begin(), BEACON_IDS.end(),
              std::back_inserter(EkfSlamTestHelper::estimate(make_in_out(ekf_slam)).beacon_ids));

    std::vector<BeaconObservation> observations;
    for (int beacon_idx = 0; beacon_idx < static_cast<int>(BEACON_IDS.size()); beacon_idx++) {
        const Eigen::Vector2d beacon_in_expected_robot =
            expected_local_from_robot.inverse() * beacons_in_local[beacon_idx];
        observations.push_back({.maybe_id = BEACON_IDS[beacon_idx],
                                .maybe_range_m = beacon_in_expected_robot.norm(),
                                .maybe_bearing_rad = std::atan2(beacon_in_expected_robot.y(),
                                                                beacon_in_expected_robot.x())});
    }

    // Action
    constexpr time::RobotTimestamp::duration DT = 100ms;
    for (int i = 0; i < 10; i++) {
        ekf_slam.predict(i * DT + time::RobotTimestamp(), liegroups::SE2());
        ekf_slam.update(observations);
    }

    const auto est = ekf_slam.estimate();

    // Verification
    EXPECT_NEAR(
        (est.local_from_robot().translation() - expected_local_from_robot.translation()).norm(),
        0.0, 1.0);
    EXPECT_NEAR(est.local_from_robot().so2().log() - expected_local_from_robot.so2().log(), 0.0,
                0.02);
}

TEST(EkfSlamTest, rotating_in_place_yields_same_pos_covariance_directions) {
    // Setup
    constexpr auto CONFIG = EkfSlamConfig{
        .max_num_beacons = 0,
        .initial_beacon_uncertainty_m = 0,
        .along_track_process_noise_m_per_rt_meter = 0.0,
        .cross_track_process_noise_m_per_rt_meter = 0.0,
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 0.0,
        .beacon_pos_process_noise_m_per_rt_s = 0.0,
        .range_measurement_noise_m = 0.0,
        .bearing_measurement_noise_rad = 0.0,
    };
    const EkfSlamEstimate est = {
        .mean = liegroups::SE2::trans(12.0, 34.0).log(),
        .cov = Eigen::DiagonalMatrix<double, 3>(1.0, 2.0, 3.0),
    };

    const liegroups::SE2 old_robot_from_new_robot = liegroups::SE2(std::numbers::pi / 2, {0, 0});

    // Action
    const EkfSlamEstimate new_est =
        detail::prediction_update(est, time::RobotTimestamp(), old_robot_from_new_robot, CONFIG);

    // Verification
    const auto cov_in_local = [](const auto &est) -> Eigen::Matrix3d {
        const Eigen::Matrix3d adjoint = est.local_from_robot().Adj();
        return adjoint.transpose() * est.robot_cov() * adjoint;
    };
    const Eigen::Matrix3d init_cov_in_local = cov_in_local(est);
    const Eigen::Matrix3d new_cov_in_local = cov_in_local(new_est);
    EXPECT_NEAR(
        (init_cov_in_local.topLeftCorner(2, 2) - new_cov_in_local.topLeftCorner(2, 2)).norm(), 0.0,
        1e-6);
}

TEST(EkfSlamTest, incorporate_mapped_landmark) {
    // Setup
    constexpr auto CONFIG = EkfSlamConfig{
        .max_num_beacons = 1,
        .initial_beacon_uncertainty_m = 100,
        .along_track_process_noise_m_per_rt_meter = 0.0,
        .cross_track_process_noise_m_per_rt_meter = 0.0,
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 0.0,
        .beacon_pos_process_noise_m_per_rt_s = 0.0,
        .range_measurement_noise_m = 0.0,
        .bearing_measurement_noise_rad = 0.0,
    };

    constexpr int BEACON_ID = 0;
    const Eigen::Vector2d BEACON_IN_LOCAL{{1.0, 2.0}};
    const Eigen::Matrix2d COV_IN_LOCAL{{{3.0, 4.0}, {3.0, 4.0}}};
    const MappedLandmarks landmarks = {
        .landmarks =
            {
                {
                    .beacon =
                        {
                            .id = BEACON_ID,
                            .pos_in_local = BEACON_IN_LOCAL,
                        },
                    .cov_in_local = COV_IN_LOCAL,
                },
            },
    };
    EkfSlam ekf_slam(CONFIG, time::RobotTimestamp());

    // Action
    const EkfSlamEstimate &est = ekf_slam.load_map(landmarks);

    // Verification
    const std::optional<Eigen::Vector2d> maybe_beacon_in_local = est.beacon_in_local(BEACON_ID);
    const std::optional<Eigen::Matrix2d> maybe_cov_in_local = est.beacon_cov(BEACON_ID);
    EXPECT_TRUE(maybe_beacon_in_local.has_value());
    EXPECT_TRUE(maybe_cov_in_local.has_value());
    EXPECT_NEAR((maybe_beacon_in_local.value() - BEACON_IN_LOCAL).norm(), 0.0, TOL);
    EXPECT_NEAR((maybe_cov_in_local.value() - COV_IN_LOCAL).norm(), 0.0, TOL);
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
    const auto [meas, pred, innovation, obs_mat] =
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
    const liegroups::SE2 est_local_from_robot;
    const Eigen::Vector2d est_beacon_123_in_local{10.0, 0.0};
    const Eigen::Vector2d est_beacon_456_in_local{0.0, 20.0};

    const EkfSlamEstimate estimate = {
        .mean = (Eigen::VectorXd(ESTIMATE_DIM) << est_local_from_robot.log(),
                 est_beacon_123_in_local, est_beacon_456_in_local)
                    .finished(),
        .cov = Eigen::MatrixXd::Zero(ESTIMATE_DIM, ESTIMATE_DIM),
        .beacon_ids = {123, 456}};

    // Action
    const auto [meas, pred, innovation, obs_mat] =
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
                    0.0, TOL);
    }
}

TEST(CreateMeasurementTest, innovation_handles_wrap_around) {
    // Setup
    constexpr int BEACON_ID = 123;
    constexpr double RANGE_M = 10.0;
    constexpr double MEASURED_BEARING_RAD = 3.0;
    constexpr double ESTIMATED_BEARING_RAD = -3.0;
    const std::vector<BeaconObservation> observations{
        {.maybe_id = BEACON_ID,
         .maybe_range_m = RANGE_M,
         .maybe_bearing_rad = MEASURED_BEARING_RAD},
    };

    const int ESTIMATE_DIM = 3 + 2 * observations.size();
    // Place the robot at the origin with the +x axes aligned. Place a beacon at (10.0, 0.0) and
    // (0.0, 20.0) in the world frame.
    // This default constructs to the identity
    const liegroups::SE2 est_local_from_robot;
    const Eigen::Vector2d est_beacon_123_in_local{RANGE_M * std::cos(ESTIMATED_BEARING_RAD),
                                                  RANGE_M * std::sin(ESTIMATED_BEARING_RAD)};

    const EkfSlamEstimate estimate = {
        .mean =
            (Eigen::VectorXd(ESTIMATE_DIM) << est_local_from_robot.log(), est_beacon_123_in_local)
                .finished(),
        .cov = Eigen::MatrixXd::Zero(ESTIMATE_DIM, ESTIMATE_DIM),
        .beacon_ids = {BEACON_ID}};

    // Action
    const auto [meas, pred, innovation, obs_mat] =
        detail::compute_measurement_and_prediction(observations, estimate);

    // Verification
    auto angle_wrap = [](const double angle_rad) {
        return std::fmod(angle_rad + std::numbers::pi, 2 * std::numbers::pi) - std::numbers::pi;
    };
    EXPECT_NEAR(innovation(1), angle_wrap(MEASURED_BEARING_RAD - ESTIMATED_BEARING_RAD), TOL);

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
