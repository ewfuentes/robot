
#pragma once

#include <optional>

#include "Eigen/Core"
#include "common/liegroups/se2.hh"
#include "experimental/beacon_sim/generate_observations.hh"

namespace robot::experimental::beacon_sim {

struct EkfSlamConfig {
    int max_num_beacons;
    double initial_beacon_uncertainty_m;
    double along_track_process_noise_m_per_rt_meter;
    double cross_track_process_noise_m_per_rt_meter;
    double heading_process_noise_rad_per_rt_meter;
    double beacon_pos_process_noise_m_per_rt_s;
    double range_measurement_noise_m;
    double bearing_measurement_noise_rad;
};

struct EkfSlamEstimate {
    // The full mean and covariance of the robot position and beacon positions
    Eigen::VectorXd mean;
    Eigen::MatrixXd cov;

    // This vector gives the beacon id of the ith landmark
    std::vector<int> beacon_ids;

    liegroups::SE2 local_from_robot() const;
    Eigen::Matrix3d robot_cov() const;

    // Returns none if beacon id estimated
    std::optional<Eigen::Vector2d> beacon_in_local(const int beacon_id) const;
    std::optional<Eigen::Matrix2d> beacon_cov(const int beacon_id) const;
};

class EkfSlam {
   public:
    explicit EkfSlam(const EkfSlamConfig &config);

    const EkfSlamEstimate &predict(const liegroups::SE2 &old_robot_from_new_robot);

    const EkfSlamEstimate &update(const std::vector<BeaconObservation> &observations);

    const EkfSlamEstimate &estimate() const { return estimate_; }

   private:
    EkfSlamConfig config_;
    EkfSlamEstimate estimate_;

    friend class EkfSlamTestHelper;
};

namespace detail {
struct UpdateInputs {
    Eigen::VectorXd measurement;
    Eigen::VectorXd prediction;
    Eigen::VectorXd innovation;
    Eigen::MatrixXd observation_matrix;
};

UpdateInputs compute_measurement_and_prediction(const std::vector<BeaconObservation> &observations,
                                                const EkfSlamEstimate &est);
}  // namespace detail
}  // namespace robot::experimental::beacon_sim