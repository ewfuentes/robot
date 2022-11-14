
#pragma once

#include "planning/probabilistic_road_map.hh"
#include "planning/belief_road_map.hh"
#include "experimental/beacon_sim/ekf_slam.hh"

namespace robot::experimental::beacon_sim {
planning::BRMPlan compute_belief_road_map_plan(const planning::RoadMap &road_map,
                                               const EkfSlam &ekf);
}
