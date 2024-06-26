
#pragma once

#include <optional>
#include <string>
#include <variant>

#include "Eigen/Core"
#include "common/liegroups/se2.hh"
#include "common/time/robot_time.hh"

namespace robot::experimental::beacon_sim {

struct NoPlannerConfig {};
struct BeliefRoadMapPlannerConfig {
    bool allow_brm_backtracking;
};
struct LandmarkBeliefRoadMapPlannerConfig {
    std::optional<int> max_num_components;
};
struct InfoLowerBoundPlannerConfig {
    double info_lower_bound_at_goal;
};

using PlannerConfig = std::variant<NoPlannerConfig, BeliefRoadMapPlannerConfig,
                                   LandmarkBeliefRoadMapPlannerConfig, InfoLowerBoundPlannerConfig>;

struct SimConfig {
    std::optional<std::string> log_path;
    std::optional<std::string> map_input_path;
    std::optional<std::string> map_output_path;
    std::optional<std::string> world_map_config;
    std::optional<std::string> road_map_config;
    Eigen::Vector2d goal_in_map;
    liegroups::SE2 map_from_initial_robot;
    time::RobotTimestamp::duration dt;
    PlannerConfig planner_config;
    bool load_off_diagonals;
    bool autostep;
    std::optional<int> correlated_beacons_configuration;
};
}  // namespace robot::experimental::beacon_sim
