
#pragma once

#include "experimental/beacon_sim/ekf_slam.hh"
#include "planning/belief_road_map.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {
struct RobotBelief {
    liegroups::SE2 local_from_robot;
    Eigen::Matrix3d cov_in_robot;
};

planning::BRMPlan<RobotBelief> compute_belief_road_map_plan(const planning::RoadMap &road_map,
                                                            const EkfSlam &ekf,
                                                            const Eigen::Vector2d &goal_state);
}  // namespace robot::experimental::beacon_sim
