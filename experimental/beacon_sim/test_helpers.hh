
#pragma once

#include <tuple>

#include "experimental/beacon_sim/ekf_slam.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {
constexpr int GRID_BEACON_ID = 123;
std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_grid_environment(
    const EkfSlamConfig &ekf_config, const double p_lone_beacon = 0.5);

std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_diamond_environment(
    const EkfSlamConfig &ekf_config, const double p_lone_beacon, const double p_no_stack_beacon,
    const double p_stacked_beacon);

std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_stress_test_environment(
    const EkfSlamConfig &ekf_config);

std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_circle_environment(
    const EkfSlamConfig &ekf_config, const int num_landmarks, const double circle_radius_m);
}  // namespace robot::experimental::beacon_sim
