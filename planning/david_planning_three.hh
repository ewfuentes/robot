#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <ctime>
#include <unordered_map>

#include "experimental/beacon_sim/beacon_potential.hh"
#include "common/math/logsumexp.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "planning/road_map.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"

namespace robot::planning {

struct DavidPlannerConfigThree {
    const int max_visits;
    const int max_plans;
    const double max_sensor_range_m;
    std::optional<time::RobotTimestamp::duration> timeout;
    std::optional<double> uncertainty_tolerance;
};

template <typename State>
struct SuccessorThree {
    State state; 
    double edge_cost;
    double weight;
};

template <typename State>
struct DavidPlannerResultThree {
    std::vector<State> nodes;
    double log_probability_mass_tracked;
};

template <typename State, typename SuccessorFunc>
std::optional<State> pick_successor_three(const State &curr_state, const SuccessorFunc &successor_for_state,
                    InOut<std::mt19937> gen, const double &favor_goal, std::unordered_map<State,int> *states,
                    const DavidPlannerConfigThree config, std::unordered_map<int,Eigen::Vector2d> *local_beacon_coords,
                    const std::vector<Eigen::Vector2d> &node_coords,
                    const RoadMap &road_map);

template <typename State, typename SuccessorFunc>
std::optional<std::vector<State>> sample_path_three(
    const DavidPlannerConfigThree &config,
    const RoadMap &road_map, InOut<std::mt19937> gen,
    const double &favor_goal, const SuccessorFunc &successor_for_state,
    const std::unordered_map<int,Eigen::Vector2d> &beacon_coords,
    const std::vector<Eigen::Vector2d> &node_coords);

template <typename State, typename SuccessorFunc>
std::optional<DavidPlannerResultThree<State>> david_planner_three( 
    const SuccessorFunc &successor_for_state,
    const RoadMap &road_map, const DavidPlannerConfigThree config,
    const experimental::beacon_sim::EkfSlamEstimate &ekf_estimate,
    const experimental::beacon_sim::BeaconPotential &beacon_potential,
    const experimental::beacon_sim::EkfSlam &ekf,const double &favor_goal);
}  // namespace robot::planning
