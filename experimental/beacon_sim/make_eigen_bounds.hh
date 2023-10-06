#pragma once 

#include <optional>
#include <unordered_set>
#include <variant>

#include "experimental/beacon_sim/make_belief_updater.hh"

namespace robot::experimental::beacon_sim {


// https://www.desmos.com/calculator/rbt2nj14ec
double step_lower_eigen_bound( // returns a lower bound on min ev of \Omega_{t-1}
    const double lower_eigen_value_information, // min ev of \Omega_t
    const double upper_eigen_value_dynamics, // max ev of G_t^{-1} G_t^{-T}
    const double upper_eigen_value_measurement, // max ev of M_t
    const double lower_eigen_value_process_noise // min ev of R_t
);

double compute_backwards_edge_belief_transform(
    const double lower_eigen_value_information, // min ev of \Omega_t
    const liegroups::SE2 &local_from_robot, 
    const Eigen::Vector2d &start_state_in_local,
    const EkfSlamConfig &ekf_config, 
    const EkfSlamEstimate &ekf_estimate,
    const std::optional<std::vector<int>> &available_beacons, 
    const double max_sensor_range_m
);


}  // namespace robot::experimental::beacon_sim
