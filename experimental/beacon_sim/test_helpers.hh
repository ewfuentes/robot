
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
//David's environment. 5x5 grid with two beacons. Independent probabilities.
/*
std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_david_indep_beacon_environment(
    const EkfSlamConfig &ekf_config, const double p_beacon_one, const double p_beacon_two);
*/
//David's environment. 5x5 grid with two beacon stacks. Independent Probabilities. Skewed number of beacons
/*
std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_david_indep_stacked_environment(
    const EkfSlamConfig &ekf_config, const double p_stack_one, const double p_no_stack_one, const double p_stack_two,const double p_no_stack_two);
*/
std::tuple<planning::RoadMap, EkfSlam, BeaconPotential> create_david_dep_stacked_environment(
    const EkfSlamConfig &ekf_config);
}  // namespace robot::experimental::beacon_sim


