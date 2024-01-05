
#pragma once

#include <optional>
#include <random>

#include "common/time/robot_time.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/world_map.hh"
#include "planning/belief_road_map.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {
struct Plan {
    time::RobotTimestamp time_of_validity;
    std::variant<planning::BRMPlan<RobotBelief>, planning::BRMPlan<LandmarkRobotBelief>> brm_plan;
};

struct Goal {
    time::RobotTimestamp time_of_validity;
    Eigen::Vector2d goal_position;
};

struct BeaconSimState {
    time::RobotTimestamp time_of_validity;
    WorldMap map;
    planning::RoadMap road_map;
    RobotState robot;
    EkfSlam ekf;
    std::vector<BeaconObservation> observations;
    std::optional<Goal> goal;
    std::optional<Plan> plan;
    std::mt19937 gen;
};
}  // namespace robot::experimental::beacon_sim
