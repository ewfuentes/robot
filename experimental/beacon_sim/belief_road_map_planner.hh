
#pragma once

#include <optional>

#include "common/liegroups/se2.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "planning/belief_road_map.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {
struct RobotBelief {
    liegroups::SE2 local_from_robot;
    Eigen::Matrix3d cov_in_robot;
};

std::optional<planning::BRMPlan<RobotBelief>> compute_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const Eigen::Vector2d &goal_state, const double max_sensor_range_m,
    const int num_start_connections, const int num_goal_connections,
    const double uncertainty_tolerance);

namespace detail {
struct ScatteringTransform {
    liegroups::SE2 local_from_robot;
    Eigen::Matrix<double, 2 * liegroups::SE2::DoF, 2 * liegroups::SE2::DoF> cov_transform;
};

ScatteringTransform compute_edge_belief_transform(const liegroups::SE2 &local_from_robot,
                                                  const Eigen::Vector2d &end_state_in_local,
                                                  const EkfSlamConfig &ekf_config,
                                                  const EkfSlamEstimate &ekf_estimate,
                                                  const BeaconPotential &beacon_potential,
                                                  const double max_sensor_range_m);
}  // namespace detail
}  // namespace robot::experimental::beacon_sim
