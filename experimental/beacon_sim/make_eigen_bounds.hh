#pragma once 

#include <optional>
#include <unordered_set>
#include <variant>

#include "experimental/beacon_sim/make_belief_updater.hh"

namespace robot::experimental::beacon_sim {


double compute_backwards_edge_belief_transform(
    const liegroups::SE2 &local_from_robot, const Eigen::Vector2d &start_state_in_local,
    const EkfSlamConfig &ekf_config, const EkfSlamEstimate &ekf_estimate,
    const std::optional<std::vector<int>> &available_beacons, const double max_sensor_range_m,
    const TransformType transform_type);


}  // namespace robot::experimental::beacon_sim
