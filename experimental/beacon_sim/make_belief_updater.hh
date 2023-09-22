
#pragma once

#include "Eigen/Core"

#include <unordered_set>

#include "common/liegroups/se2.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "planning/probabilistic_road_map.hh"
#include "planning/belief_road_map.hh"

namespace robot::experimental::beacon_sim {
struct EdgeTransform {
    static constexpr int DIM = 2 * liegroups::SE2::DoF;
    using Matrix = Eigen::Matrix<double, DIM, DIM>;
    liegroups::SE2 local_from_robot;
    std::vector<double> weight;
    std::vector<Matrix> transforms;
};

std::tuple<liegroups::SE2, EdgeTransform::Matrix> compute_edge_belief_transform(
    const liegroups::SE2 &local_from_robot, const Eigen::Vector2d &end_state_in_local,
    const EkfSlamConfig &ekf_config, const EkfSlamEstimate &ekf_estimate,
    const std::optional<std::vector<int>> &available_beacons, const double max_sensor_range_m);

EdgeTransform compute_edge_belief_transform(const liegroups::SE2 &local_from_robot,
                                            const Eigen::Vector2d &end_state_in_local,
                                            const EkfSlamConfig &ekf_config,
                                            const EkfSlamEstimate &ekf_estimate,
                                            const BeaconPotential &beacon_potential,
                                            const double max_sensor_range_m,
                                            const int max_num_transforms);

planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const Eigen::Vector2d &goal_state,
                                                         const double max_sensor_range_m,
                                                         const int max_num_transforms,
                                                         const EkfSlam &ekf,
                                                         const BeaconPotential &beacon_potential);

planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const Eigen::Vector2d &goal_state,
                                                         const double max_sensor_range_m,
                                                         const EkfSlam &ekf,
                                                         const std::vector<int> &present_beacons);

}
