
#include "experimental/beacon_sim/ekf_slam.hh"

#include <iostream>

namespace robot::experimental::beacon_sim {
namespace {

constexpr int ROBOT_STATE_DIM = liegroups::SE2::DoF;
constexpr int BEACON_DIM = 2;

std::optional<int> find_beacon_matrix_idx(const std::vector<int> &ids, const int id) {
    const auto iter = std::find(ids.begin(), ids.end(), id);
    if (iter == ids.end()) {
        return std::nullopt;
    }
    // This is the index of the beacon of interest.
    const int beacon_idx = std::distance(ids.begin(), iter);
    // Next we transform from a beacon index to an entry in the mean/cov
    return ROBOT_STATE_DIM + BEACON_DIM * beacon_idx;
}

std::optional<int> find_beacon_matrix_idx_or_add(const int id, InOut<EkfSlamEstimate> est) {
    std::optional<int> maybe_idx = find_beacon_matrix_idx(est->beacon_ids, id);
    if (maybe_idx.has_value()) {
        return maybe_idx.value();
    }

    const int max_num_landmarks = (est->mean.rows() - ROBOT_STATE_DIM) / BEACON_DIM;
    if (max_num_landmarks <= static_cast<int>(est->beacon_ids.size())) {
        return std::nullopt;
    }

    est->beacon_ids.push_back(id);
    return find_beacon_matrix_idx(est->beacon_ids, id);
}

}  // namespace

namespace detail {
UpdateInputs compute_measurement_and_prediction(
    const std::vector<BeaconObservation> &observations, const EkfSlamEstimate &est,
    const std::optional<liegroups::SE2> &maybe_local_from_robot) {
    // Preallocate space for the maximum possible size
    Eigen::VectorXd measurement_vec = Eigen::VectorXd::Zero(observations.size() * 2);
    Eigen::VectorXd prediction_vec = Eigen::VectorXd::Zero(observations.size() * 2);
    Eigen::VectorXd innovation_vec = Eigen::VectorXd::Zero(observations.size() * 2);
    Eigen::MatrixXd observation_mat =
        Eigen::MatrixXd::Zero(observations.size() * 2, est.mean.rows());

    int num_valid_observations = 0;
    const auto est_local_from_robot = maybe_local_from_robot.has_value()
                                          ? maybe_local_from_robot.value()
                                          : est.local_from_robot();
    for (const auto &obs : observations) {
        const bool is_complete_observation = obs.maybe_id.has_value() &&
                                             obs.maybe_range_m.has_value() &&
                                             obs.maybe_bearing_rad.has_value();
        if (!is_complete_observation) {
            continue;
        }
        // Populate the observation vector
        // The entries are [range_a bearing_a range_b bearing_b]
        const int start_idx = 2 * num_valid_observations;
        measurement_vec(start_idx) = obs.maybe_range_m.value();
        measurement_vec(start_idx + 1) = obs.maybe_bearing_rad.value();

        // Populate the prediction vector
        const Eigen::Vector2d est_beacon_in_local =
            est.beacon_in_local(obs.maybe_id.value()).value();
        const Eigen::Vector2d est_beacon_in_robot =
            est_local_from_robot.inverse() * est_beacon_in_local;

        if (est_beacon_in_robot.norm() < 0.1) {
            // We're too close, so we won't get a good linearization on heading
            // Skip for now. Since start_idx has not been incremented, the next
            // valid measurement will be written in the same location, if no
            // valid measurements are remaining, the measurement vector will be
            // appropriately resized.
            continue;
        }

        prediction_vec(start_idx) = est_beacon_in_robot.norm();
        prediction_vec(start_idx + 1) =
            std::atan2(est_beacon_in_robot.y(), est_beacon_in_robot.x());

        const liegroups::SO2 robot_from_measured(obs.maybe_bearing_rad.value());
        const liegroups::SO2 robot_from_predicted(prediction_vec(start_idx + 1));

        innovation_vec(start_idx) = measurement_vec(start_idx) - prediction_vec(start_idx);
        innovation_vec(start_idx + 1) =
            (robot_from_predicted.inverse() * robot_from_measured).log();

        // Populate the measurement matrix

        // let X = local_from_robot
        // let pl = beacon_in_local
        // let pr = beacon_in_robot
        // pr = X.inverse() * pl
        // Using right jacobians
        // d_pr_d_X = d_pr_d_X^-1 * dX^-1_dX
        // d_pr_d_X = [R.t R.t[1]_x * pl] * -Ad_x
        // d_pr_d_X = [-I R.t*[1]*(t - pl)]
        // where [1] is the generator for so2
        // d_pr_d_pl = R.t
        const Eigen::Matrix2d R = est_local_from_robot.so2().matrix();
        const Eigen::Vector2d t = est_local_from_robot.translation();
        const Eigen::Matrix<double, 2, 3> d_pr_d_X = [&]() {
            Eigen::Matrix<double, 2, 3> out;
            out << -Eigen::Matrix2d::Identity(),
                R.transpose() * liegroups::SO2::generator() * (t - est_beacon_in_local);
            return out;
        }();
        const Eigen::Matrix2d d_pr_d_pl = R.transpose();

        // range = sqrt(beacon_in_robot.dot(beacon_in_robot))
        // d_range_d_X = drange_d_pr * d_pr_d_X
        // let u = pr.dot(pr)
        // d_range_d_X = d_range_d_u * d_u_d_pr * d_pr_d_X
        // d_range_d_u = 0.5 / sqrt(u)
        // d_u_d_pr = 2 * pr.T
        // dims = 1x1 1x2 2x3 = 1x3
        const double est_sq_range_m2 = est_beacon_in_robot.squaredNorm();
        const double est_range_m = std::sqrt(est_sq_range_m2);
        const double d_range_d_u = 0.5 / est_range_m;
        const Eigen::RowVector2d d_u_d_pr = 2.0 * est_beacon_in_robot.transpose();
        const Eigen::RowVector3d d_range_d_X = d_range_d_u * d_u_d_pr * d_pr_d_X;

        // d_range_d_pl = d_range_d_u * d_u_d_pr * d_pr_d_pl
        // dims = 1x1 1x2 2x2 = 1x2
        const Eigen::RowVector2d d_range_d_pl = d_range_d_u * d_u_d_pr * d_pr_d_pl;

        // bearing = atan2(beacon_in_robot.y(), beacon_in_robot.x())
        // d_bearing_d_X = d_bearing_d_pr * d_pr_d_X
        // d_bearing_d_pr = (-y / (range * range), x / (range * range))
        // dims = 1x2 2x3 = 1x3

        const Eigen::RowVector2d d_bearing_d_pr{-est_beacon_in_robot.y() / est_sq_range_m2,
                                                est_beacon_in_robot.x() / est_sq_range_m2};
        const Eigen::RowVector3d d_bearing_d_X = d_bearing_d_pr * d_pr_d_X;

        // dbearing_d_pl = dbearing_d_pr * d_pr_d_pl
        // dims = 1x2 2x2 = 1x2
        const Eigen::RowVector2d d_bearing_d_pl = d_bearing_d_pr * d_pr_d_pl;

        auto build_observation_row =
            [&est, &obs](const Eigen::RowVector3d &d_obs_d_X,
                         const Eigen::RowVector2d &d_obs_d_pl) -> Eigen::RowVectorXd {
            Eigen::RowVectorXd out = Eigen::RowVectorXd::Zero(est.mean.rows());
            out(Eigen::seqN(0, ROBOT_STATE_DIM)) = d_obs_d_X;
            const int beacon_idx =
                find_beacon_matrix_idx(est.beacon_ids, obs.maybe_id.value()).value();
            out(Eigen::seqN(beacon_idx, BEACON_DIM)) = d_obs_d_pl;
            return out;
        };

        Eigen::RowVectorXd linear_range_obs = build_observation_row(d_range_d_X, d_range_d_pl);
        Eigen::RowVectorXd linear_bearing_obs =
            build_observation_row(d_bearing_d_X, d_bearing_d_pl);

        observation_mat.row(start_idx) = linear_range_obs;
        observation_mat.row(start_idx + 1) = linear_bearing_obs;

        num_valid_observations++;
    }

    // Shrink the measurement vector and observation matrix to the number of valid rows
    const int expected_size = num_valid_observations * BEACON_DIM;
    measurement_vec.conservativeResize(expected_size);
    prediction_vec.conservativeResize(expected_size);
    innovation_vec.conservativeResize(expected_size);
    observation_mat.conservativeResize(expected_size, Eigen::NoChange_t{});

    return {.measurement = measurement_vec,
            .prediction = prediction_vec,
            .innovation = innovation_vec,
            .observation_matrix = observation_mat};
}

EkfSlamEstimate prediction_update(const EkfSlamEstimate &est, const time::RobotTimestamp &time,
                                  const liegroups::SE2 &old_robot_from_new_robot,
                                  const EkfSlamConfig &config) {
    // Update the robot mean
    EkfSlamEstimate out = est;
    out.time_of_validity = time;
    const auto local_from_new_robot = est.local_from_robot() * old_robot_from_new_robot;
    out.mean(Eigen::seqN(0, ROBOT_STATE_DIM)) =
        (Eigen::Vector3d() << local_from_new_robot.translation(), local_from_new_robot.so2().log())
            .finished();

    // Update the covariances
    const std::chrono::duration<double> dt_s = (time - est.time_of_validity);
    const Eigen::Matrix3d robot_process_noise = Eigen::DiagonalMatrix<double, ROBOT_STATE_DIM>(
        config.along_track_process_noise_m_per_rt_meter *
                config.along_track_process_noise_m_per_rt_meter *
                old_robot_from_new_robot.arclength() +
            config.pos_process_noise_m_per_rt_s * config.pos_process_noise_m_per_rt_s *
                dt_s.count(),
        config.cross_track_process_noise_m_per_rt_meter *
                config.cross_track_process_noise_m_per_rt_meter *
                old_robot_from_new_robot.arclength() +
            config.pos_process_noise_m_per_rt_s * config.pos_process_noise_m_per_rt_s *
                dt_s.count(),
        config.heading_process_noise_rad_per_rt_meter *
                config.heading_process_noise_rad_per_rt_meter *
                old_robot_from_new_robot.arclength() +
            config.heading_process_noise_rad_per_rt_s * config.heading_process_noise_rad_per_rt_s *
                dt_s.count());

    const Eigen::Matrix3d dynamics_jac_wrt_state = old_robot_from_new_robot.inverse().Adj();
    std::cout << "Dynamics wrt Jac:" << std::endl << dynamics_jac_wrt_state << std::endl;
    out.cov.block(0, 0, ROBOT_STATE_DIM, ROBOT_STATE_DIM) =
        dynamics_jac_wrt_state * out.cov.block(0, 0, ROBOT_STATE_DIM, ROBOT_STATE_DIM) *
            dynamics_jac_wrt_state.transpose() +
        robot_process_noise;

    const int num_landmark_cov_cols = out.cov.cols() - ROBOT_STATE_DIM;
    out.cov.topRightCorner(ROBOT_STATE_DIM, num_landmark_cov_cols) =
        dynamics_jac_wrt_state * out.cov.topRightCorner(ROBOT_STATE_DIM, num_landmark_cov_cols);
    out.cov.bottomLeftCorner(num_landmark_cov_cols, ROBOT_STATE_DIM) =
        out.cov.bottomLeftCorner(num_landmark_cov_cols, ROBOT_STATE_DIM) *
        dynamics_jac_wrt_state.transpose();

    const Eigen::MatrixXd landmark_pos_process_noise =
        Eigen::MatrixXd::Identity(num_landmark_cov_cols, num_landmark_cov_cols) *
        config.beacon_pos_process_noise_m_per_rt_s * config.beacon_pos_process_noise_m_per_rt_s *
        dt_s.count();

    out.cov.bottomRightCorner(num_landmark_cov_cols, num_landmark_cov_cols) +=
        landmark_pos_process_noise;

    return out;
}

EkfSlamEstimate incorporate_mapped_landmarks(const EkfSlamEstimate &est,
                                             const MappedLandmarks &landmarks,
                                             const bool should_load_off_diagonals) {
    EkfSlamEstimate out = est;
    for (int i = 0; i < static_cast<int>(landmarks.beacon_ids.size()); i++) {
        const auto maybe_matrix_idx =
            find_beacon_matrix_idx_or_add(landmarks.beacon_ids.at(i), make_in_out(out));
        if (!maybe_matrix_idx.has_value()) {
            continue;
        }
        const auto &idx = maybe_matrix_idx.value();

        // Copy the mean
        out.mean(Eigen::seqN(idx, BEACON_DIM)) = landmarks.beacon_in_local.at(i);

        // Copy the main diagonal block
        out.cov.block(idx, idx, BEACON_DIM, BEACON_DIM) =
            landmarks.cov_in_local.block(BEACON_DIM * i, BEACON_DIM * i, BEACON_DIM, BEACON_DIM);

        if (!should_load_off_diagonals) {
            continue;
        }

        for (int j = i + 1; j < static_cast<int>(landmarks.beacon_ids.size()); j++) {
            const auto maybe_other_matrix_idx =
                find_beacon_matrix_idx_or_add(landmarks.beacon_ids.at(j), make_in_out(out));
            if (!maybe_other_matrix_idx.has_value()) {
                continue;
            }
            const auto &other_idx = maybe_other_matrix_idx.value();
            // Copy the i, j off diagonal blocks
            out.cov.block(idx, other_idx, BEACON_DIM, BEACON_DIM) = landmarks.cov_in_local.block(
                BEACON_DIM * i, BEACON_DIM * j, BEACON_DIM, BEACON_DIM);

            out.cov.block(other_idx, idx, BEACON_DIM, BEACON_DIM) = landmarks.cov_in_local.block(
                BEACON_DIM * j, BEACON_DIM * i, BEACON_DIM, BEACON_DIM);
        }
    }
    return out;
}
}  // namespace detail

liegroups::SE2 EkfSlamEstimate::local_from_robot() const {
    return liegroups::SE2(mean(2), mean.head<2>());
}

void EkfSlamEstimate::local_from_robot(const liegroups::SE2 &local_from_robot) {
    mean(2) = local_from_robot.so2().log();
    mean.head<2>() = local_from_robot.translation();
}

Eigen::Matrix3d EkfSlamEstimate::robot_cov() const {
    return cov.block(0, 0, ROBOT_STATE_DIM, ROBOT_STATE_DIM);
}

std::optional<Eigen::Vector2d> EkfSlamEstimate::beacon_in_local(const int beacon_id) const {
    const auto maybe_beacon_idx = find_beacon_matrix_idx(beacon_ids, beacon_id);
    if (!maybe_beacon_idx.has_value()) {
        return std::nullopt;
    }

    return mean(Eigen::seqN(maybe_beacon_idx.value(), BEACON_DIM));
}

std::optional<Eigen::Matrix2d> EkfSlamEstimate::beacon_cov(const int beacon_id) const {
    const auto maybe_beacon_idx = find_beacon_matrix_idx(beacon_ids, beacon_id);
    if (!maybe_beacon_idx.has_value()) {
        return std::nullopt;
    }
    const int beacon_idx = maybe_beacon_idx.value();
    return cov.block(beacon_idx, beacon_idx, BEACON_DIM, BEACON_DIM);
}

EkfSlam::EkfSlam(const EkfSlamConfig &config, const time::RobotTimestamp &time) : config_(config) {
    const int total_dim = ROBOT_STATE_DIM + BEACON_DIM * config_.max_num_beacons;
    estimate_.time_of_validity = time;
    estimate_.mean = Eigen::VectorXd::Zero(total_dim);
    // Initialize the beacon positions to be somewhere other than on top of the robot at init time
    estimate_.mean(Eigen::seq(ROBOT_STATE_DIM, Eigen::last)).array() += 30.0;
    estimate_.cov = Eigen::MatrixXd::Identity(total_dim, total_dim) *
                    config_.initial_beacon_uncertainty_m * config_.initial_beacon_uncertainty_m;
    estimate_.cov.block(0, 0, ROBOT_STATE_DIM, ROBOT_STATE_DIM) = Eigen::Matrix3d::Zero();
}

const EkfSlamEstimate &EkfSlam::predict(const time::RobotTimestamp &time,
                                        const liegroups::SE2 &old_robot_from_new_robot) {
    estimate_ = detail::prediction_update(estimate_, time, old_robot_from_new_robot, config_);
    return estimate_;
}

const EkfSlamEstimate &EkfSlam::load_map(const MappedLandmarks &landmarks,
                                         const bool should_load_off_diagonals) {
    estimate_ =
        detail::incorporate_mapped_landmarks(estimate_, landmarks, should_load_off_diagonals);
    // Set the ego uncertainty to be really large
    estimate_.cov(0, 0) =
        config_.on_map_load_position_uncertainty_m * config_.on_map_load_position_uncertainty_m;
    estimate_.cov(1, 1) = estimate_.cov(0, 0);
    estimate_.cov(2, 2) =
        config_.on_map_load_heading_uncertainty_rad * config_.on_map_load_heading_uncertainty_rad;
    return estimate_;
}

const EkfSlamEstimate &EkfSlam::update(const std::vector<BeaconObservation> &observations) {
    // Initialize any new beacons we may have observed
    for (const auto &obs : observations) {
        if (!(obs.maybe_id.has_value() && obs.maybe_range_m.has_value() &&
              obs.maybe_bearing_rad.has_value())) {
            // Skip partial observations
            continue;
        }
        const auto maybe_idx = find_beacon_matrix_idx(estimate_.beacon_ids, obs.maybe_id.value());
        if (maybe_idx.has_value()) {
            continue;
        }
        if (static_cast<int>(estimate_.beacon_ids.size()) >= config_.max_num_beacons) {
            // We have seen too many beacons, refuse to add it
            continue;
        }
        estimate_.beacon_ids.push_back(obs.maybe_id.value());
        const int beacon_start_idx =
            find_beacon_matrix_idx(estimate_.beacon_ids, obs.maybe_id.value()).value();
        // initialize the mean to be where the observation places it
        const double &range_m = obs.maybe_range_m.value();
        const double &bearing_rad = obs.maybe_bearing_rad.value();
        const Eigen::Vector2d beacon_in_robot{range_m * std::cos(bearing_rad),
                                              range_m * std::sin(bearing_rad)};

        const Eigen::Vector2d beacon_in_local = estimate_.local_from_robot() * beacon_in_robot;
        estimate_.mean(Eigen::seqN(beacon_start_idx, BEACON_DIM)) = beacon_in_local;
        estimate_.cov.block(beacon_start_idx, beacon_start_idx, BEACON_DIM, BEACON_DIM) =
            Eigen::Matrix<double, BEACON_DIM, BEACON_DIM>::Identity() *
            config_.initial_beacon_uncertainty_m * config_.initial_beacon_uncertainty_m;
    }

    // turn the observations into a column vector of beacon in robot points
    const auto [measurement_vec, prediction_vec, innovation, observation_mat] =
        detail::compute_measurement_and_prediction(observations, estimate_);

    // Compute the innovation
    const Eigen::MatrixXd observation_noise = [&, measurement_dim = measurement_vec.rows()]() {
        Eigen::MatrixXd noise = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim);
        // Set the even rows to be the range noise
        noise(Eigen::seq(0, Eigen::last, 2), Eigen::all) *=
            config_.range_measurement_noise_m * config_.range_measurement_noise_m;
        // Set the odd rows to be the bearing noise
        noise(Eigen::seq(1, Eigen::last, 2), Eigen::all) *=
            config_.bearing_measurement_noise_rad * config_.bearing_measurement_noise_rad;
        return noise;
    }();

    const Eigen::MatrixXd innovation_cov =
        observation_mat * estimate_.cov * observation_mat.transpose() + observation_noise;

    // Use a solve instead of an inverse
    const Eigen::MatrixXd kalman_gain =
        estimate_.cov * observation_mat.transpose() * innovation_cov.inverse();
    // Break up the mean update. Perform an exp on the robot state components
    const Eigen::VectorXd update_vector = kalman_gain * innovation;

    const liegroups::SE2 local_from_updated_robot =
        estimate_.local_from_robot() * liegroups::SE2::exp(update_vector.head<ROBOT_STATE_DIM>());

    estimate_.mean.head<ROBOT_STATE_DIM>() =
        (Eigen::Vector3d() << local_from_updated_robot.translation(),
         local_from_updated_robot.so2().log())
            .finished();

    estimate_.mean(Eigen::seq(ROBOT_STATE_DIM, Eigen::last)) +=
        update_vector(Eigen::seq(ROBOT_STATE_DIM, Eigen::last));

    const Eigen::MatrixXd I_min_KH =
        (Eigen::MatrixXd::Identity(estimate_.cov.rows(), estimate_.cov.cols()) -
         kalman_gain * observation_mat);
    estimate_.cov = I_min_KH * estimate_.cov;

    return estimate_;
}
}  // namespace robot::experimental::beacon_sim
