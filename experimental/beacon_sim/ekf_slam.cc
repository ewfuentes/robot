
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

std::tuple<Eigen::VectorXd, Eigen::MatrixXd> compute_measurement_and_observation_matrix(
    const std::vector<BeaconObservation> &obs, const EkfSlamEstimate &est) {
}
}  // namespace
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
    int num_valid_obs = 0;
    for (const auto &obs : observations) {
        if (!(obs.maybe_id.has_value() && obs.maybe_range_m.has_value() &&
              obs.maybe_bearing_rad.has_value())) {
            // Skip partial observations
            continue;
        }
        const auto maybe_idx = find_beacon_matrix_idx(estimate_.beacon_ids, obs.maybe_id.value());
        if (maybe_idx.has_value()) {
            num_valid_obs++;
            continue;
        }
        if (static_cast<int>(estimate_.beacon_ids.size()) >= config_.max_num_beacons) {
            // We have seen too many beacons, refuse to add it
            continue;
        }
        estimate_.beacon_ids.push_back(obs.maybe_id.value());
        num_valid_obs++;
    }
    // turn the observations into a column vector of beacon in robot points

    // Compute the innovation

    return estimate_;
}
}  // namespace robot::experimental::beacon_sim
