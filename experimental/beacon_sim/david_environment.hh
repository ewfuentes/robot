#pragma once

#include <tuple>

#include "experimental/beacon_sim/ekf_slam.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {
    
    std::tuple<planning::RoadMap, EkfSlam> create_david_grid_environment(
    const EkfSlamConfig &ekf_config);

    std::tuple<planning::RoadMap, EkfSlam> create_david_diamond_environment(
    const EkfSlamConfig &ekf_config);
} // namespace robot::experimental::beacon_sim