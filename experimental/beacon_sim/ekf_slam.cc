
#include "experimental/beacon_sim/ekf_slam.hh"

#include <iostream>

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
std::tuple<Eigen::VectorXd, Eigen::MatrixXd> compute_measurement_vector_and_observation_matrix(
    const std::vector<BeaconObservation> &observations, const EkfSlamEstimate &est) {
    // Preallocate space for the maximum possible size
    Eigen::VectorXd measurement_vec = Eigen::VectorXd::Zero(observations.size() * 2);
    Eigen::MatrixXd observation_mat =
        Eigen::MatrixXd::Zero(observations.size() * 2, est.mean.rows());

    int num_valid_observations = 0;
    const auto local_from_robot = est.local_from_robot();
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

        // Populate the measurement vector

        // let X = local_from_robot
        // let pl = beacon_in_local
        // let pr = beacon_in_robot
        // p_r = X.inverse() * p_l
        // Using right jacobians
        // d_pr_d_X = d_pr_d_X^-1 * dX^-1_dX
        // d_pr_d_X = [R.t R.t[1]_x * pl] * -Ad_x
        // d_pr_d_X = [-I R.t[1]_x(t - pl)]
        // d_pr_d_pl = R.t
        const Eigen::Vector2d beacon_in_local = est.beacon_in_local(obs.maybe_id.value()).value();
        const Eigen::Matrix2d R = local_from_robot.so2().matrix();
        const Eigen::Vector2d t = local_from_robot.translation();
        const Eigen::Matrix<double, 2, 3> d_beacon_in_robot_d_robot_in_local = [&]() {
            Eigen::Matrix<double, 2, 3> out;
            out << Eigen::Matrix2d::Identity(),
                R.transpose() * Sophus::SO2d::generator() * (t - beacon_in_local);
            return out;
        }();
        const Eigen::Matrix2d d_beacon_in_robot_d_beacon_in_local = R.transpose();

        // range = sqrt(beacon_in_robot.dot(beacon_in_robot))
        // d_range_d_X = drange_d_pr * d_pr_d_X
        // let u = pr.dot(pr)
        // d_range_d_X = d_range_d_u * d_u_d_pr * d_pr_d_X
        // d_range_d_u = 0.5 / sqrt(u)
        // d_u_d_pr = 2 * pr.T
        // dims = 1x1 1x2 2x3 = 1x3

        // d_range_d_pl = d_range_d_u * d_u_d_pr * d_pr_d_pl
        // dims = 1x1 1x2 2x2 = 1x2

        // bearing = atan2(beacon_in_robot.y(), beacon_in_robot.x())
        // dbearing_d_X = dbearing_d_pr * d_pr_d_X
        // dbearing_d_pr = (-y / (range * range), x / (range * range))
        // dims = 1x2 2x3 = 1x3

        // dbearing_d_pl = dbearing_d_pr * d_pr_d_pl
        // dims = 1x2 2x2 = 1x2

        (void)d_beacon_in_robot_d_robot_in_local;
        (void)d_beacon_in_robot_d_beacon_in_local;

        Eigen::RowVectorXd linear_range_obs = Eigen::RowVectorXd::Zero(est.mean.rows());
        Eigen::RowVectorXd linear_bearing_obs = Eigen::RowVectorXd::Zero(est.mean.rows());

        observation_mat.row(start_idx) = linear_range_obs;
        observation_mat.row(start_idx + 1) = linear_bearing_obs;

        num_valid_observations++;
    }

    // Shrink the measurement vector and observation matrix to the number of valid rows
    measurement_vec.conservativeResize(num_valid_observations * 2);
    observation_mat.conservativeResize(num_valid_observations * 2, Eigen::NoChange_t{});

    return std::make_tuple(measurement_vec, observation_mat);
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
    estimate_.cov = Eigen::MatrixXd::Identity(total_dim, total_dim) *
                    config_.initial_beacon_uncertainty_m * config_.initial_beacon_uncertainty_m;
    estimate_.cov.block(0, 0, ROBOT_STATE_DIM, ROBOT_STATE_DIM) = Eigen::Matrix3d::Zero();
}

const EkfSlamEstimate &EkfSlam::predict(const Sophus::SE2d &old_robot_from_new_robot) {
    // Update the robot mean
    // Since these are perturbations on the right, they can be directly added
    estimate_.mean(Eigen::seqN(0, ROBOT_STATE_DIM)) += old_robot_from_new_robot.log();

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
    const auto [measurement_vec, observation_mat] =
        detail::compute_measurement_vector_and_observation_matrix(observations, estimate_);

    // Compute the innovation

    return estimate_;
}
}  // namespace robot::experimental::beacon_sim
