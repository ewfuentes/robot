
#pragma once

#include "experimental/beacon_sim/ekf_slam.hh"
#include "planning/belief_road_map.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {
planning::BRMPlan compute_belief_road_map_plan(const planning::RoadMap &road_map,
                                               const EkfSlam &ekf);
}
