
#include "experimental/beacon_sim/ekf_slam.hh"

namespace robot::experimental::beacon_sim {
namespace {

constexpr int ROBOT_STATE_DIM = Sophus::SE2d::DoF;
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

}  // namespace

namespace detail {
UpdateInputs compute_measurement_and_prediction(const std::vector<BeaconObservation> &observations,
                                                const EkfSlamEstimate &est) {
    // Preallocate space for the maximum possible size
    Eigen::VectorXd measurement_vec = Eigen::VectorXd::Zero(observations.size() * 2);
    Eigen::VectorXd prediction_vec = Eigen::VectorXd::Zero(observations.size() * 2);
    Eigen::VectorXd innovation_vec = Eigen::VectorXd::Zero(observations.size() * 2);
    Eigen::MatrixXd observation_mat =
        Eigen::MatrixXd::Zero(observations.size() * 2, est.mean.rows());

    int num_valid_observations = 0;
    const auto est_local_from_robot = est.local_from_robot();
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

        prediction_vec(start_idx) = est_beacon_in_robot.norm();
        prediction_vec(start_idx + 1) =
            std::atan2(est_beacon_in_robot.y(), est_beacon_in_robot.x());

        const Sophus::SO2 robot_from_measured(obs.maybe_bearing_rad.value());
        const Sophus::SO2 robot_from_predicted(prediction_vec(start_idx + 1));

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
                R.transpose() * Sophus::SO2d::generator() * (t - est_beacon_in_local);
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
}  // namespace detail

Sophus::SE2d EkfSlamEstimate::local_from_robot() const {
    return Sophus::SE2d::exp(mean(Eigen::seqN(0, ROBOT_STATE_DIM)));
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

EkfSlam::EkfSlam(const EkfSlamConfig &config) : config_(config) {
    const int total_dim = ROBOT_STATE_DIM + BEACON_DIM * config_.max_num_beacons;
    estimate_.mean = Eigen::VectorXd::Zero(total_dim);
    // Initialize the beacon positions to be somewhere other than on top of the robot at init time
    estimate_.mean(Eigen::seq(ROBOT_STATE_DIM, Eigen::last)).array() += 30.0;
    estimate_.cov = Eigen::MatrixXd::Identity(total_dim, total_dim) *
                    config_.initial_beacon_uncertainty_m * config_.initial_beacon_uncertainty_m;
    estimate_.cov.block(0, 0, ROBOT_STATE_DIM, ROBOT_STATE_DIM) = Eigen::Matrix3d::Zero();
}

const EkfSlamEstimate &EkfSlam::predict(const Sophus::SE2d &old_robot_from_new_robot) {
    // Update the robot mean
    const auto local_from_new_robot = (estimate_.local_from_robot() * old_robot_from_new_robot);
    estimate_.mean(Eigen::seqN(0, ROBOT_STATE_DIM)) = local_from_new_robot.log();

    // Update the covariances
    // TODO: This should use the length of the geodesic between old and new poses
    // TODO: These are perturbations in the robot frame. I think this is correct...
    estimate_.cov.block(0, 0, ROBOT_STATE_DIM, ROBOT_STATE_DIM) +=
        Eigen::DiagonalMatrix<double, ROBOT_STATE_DIM>(
            config_.along_track_process_noise_m_per_rt_meter *
                config_.along_track_process_noise_m_per_rt_meter,
            config_.cross_track_process_noise_m_per_rt_meter *
                config_.cross_track_process_noise_m_per_rt_meter,
            config_.heading_process_noise_rad_per_rt_meter *
                config_.heading_process_noise_rad_per_rt_meter);

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
            config_.bearing_measurement_noise_rad * config_.range_measurement_noise_m;
        return noise;
    }();

    const Eigen::MatrixXd innovation_cov =
        observation_mat * estimate_.cov * observation_mat.transpose() + observation_noise;

    // Use a solve instead of an inverse
    const Eigen::MatrixXd kalman_gain =
        estimate_.cov * observation_mat.transpose() * innovation_cov.inverse();
    estimate_.mean += kalman_gain * innovation;
    estimate_.cov = (Eigen::MatrixXd::Identity(estimate_.cov.rows(), estimate_.cov.cols()) -
                     kalman_gain * observation_mat) *
                    estimate_.cov;

    return estimate_;
}
}  // namespace robot::experimental::beacon_sim
