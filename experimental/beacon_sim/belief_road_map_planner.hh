
#pragma once

#include <optional>

#include "common/liegroups/se2.hh"
#include "common/time/robot_time.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "planning/belief_road_map.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {
struct BeliefRoadMapOptions {
    double max_sensor_range_m;
    std::optional<double> uncertainty_tolerance;
    // This config controls when we switch from exact to approximate edge transforms.
    // If the number of landmark presence configurations is less than or equal this amount,
    // we compute the expectation exactly. If the number of configurations are greater than this,
    // we sample up to this many configurations and compute the sample expectation. Note that once
    // we sample the configurations for an edge, they are reused for all future edge traversals.
    int max_num_edge_transforms;
    std::optional<time::RobotTimestamp::duration> timeout;
};

struct PathConstrainedBeliefRoadMapOptions {
    // The algorithm requires looking over all possible paths from the start to the goal. To make
    // the problem tractable, we wil only consider paths that are within a multiplicative factor
    // of the shortest path.
    double max_path_length_ratio;

    double max_sensor_range_m;
};

struct LandmarkBeliefRoadMapOptions {
    struct SampledBeliefOptions {
        int max_num_components;
        int seed;
    };
    double max_sensor_range_m;
    std::optional<SampledBeliefOptions> sampled_belief_options;
    std::optional<time::RobotTimestamp::duration> timeout;
};

struct PathConstrainedBeliefPlanResult {
    std::vector<int> plan;
    Eigen::Matrix3d expected_cov;
};

struct ExpectedBeliefPlanResult {
    std::vector<int> plan;
    Eigen::Matrix3d expected_cov;
};

struct ExpectedBeliefRoadMapOptions {
    int num_landmark_configuration_samples;
    int seed;
};

std::optional<planning::BRMPlan<LandmarkRobotBelief>> compute_landmark_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const LandmarkBeliefRoadMapOptions &options);

std::optional<planning::BRMPlan<RobotBelief>> compute_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const BeliefRoadMapOptions &options);

PathConstrainedBeliefPlanResult compute_path_constrained_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const PathConstrainedBeliefRoadMapOptions &options);

std::optional<ExpectedBeliefPlanResult> compute_expected_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const ExpectedBeliefRoadMapOptions &options);

}  // namespace robot::experimental::beacon_sim
