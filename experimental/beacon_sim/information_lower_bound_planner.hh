
#pragma once

#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "planning/belief_road_map.hh"

namespace robot::experimental::beacon_sim {

planning::BRMPlan<RobotBelief> compute_info_lower_bound_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf,
    const double information_lower_bound_at_goal, const double max_sensor_range_m);

}
