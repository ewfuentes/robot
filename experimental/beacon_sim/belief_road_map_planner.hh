
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

struct BeliefRoadMapOptions {
    double max_sensor_range_m;
    int num_start_connections;
    int num_goal_connections;
    double uncertainty_tolerance;
    // This config controls when we switch from exact to approximate edge transforms.
    // If the number of landmark presence configurations is less than or equal this amount,
    // we compute the expectation exactly. If the number of configurations are greater than this,
    // we sample up to this many configurations and compute the sample expectation. Note that once
    // we sample the configurations for an edge, they are reused for all future edge traversals.
    int max_num_edge_transforms;
};

std::optional<planning::BRMPlan<RobotBelief>> compute_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const Eigen::Vector2d &goal_state, const BeliefRoadMapOptions &options);

namespace detail {
struct EdgeTransform {
    static constexpr int DIM = 2 * liegroups::SE2::DoF;
    using Matrix = Eigen::Matrix<double, DIM, DIM>;
    liegroups::SE2 local_from_robot;
    std::vector<double> weight;
    std::vector<Matrix> transforms;
};

EdgeTransform compute_edge_belief_transform(const liegroups::SE2 &local_from_robot,
                                            const Eigen::Vector2d &end_state_in_local,
                                            const EkfSlamConfig &ekf_config,
                                            const EkfSlamEstimate &ekf_estimate,
                                            const BeaconPotential &beacon_potential,
                                            const double max_sensor_range_m,
                                            const int max_num_transforms);
}  // namespace detail
}  // namespace robot::experimental::beacon_sim
