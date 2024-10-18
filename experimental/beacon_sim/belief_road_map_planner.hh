
#pragma once

#include <optional>

#include "common/liegroups/se2.hh"
#include "common/time/robot_time.hh"
#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/make_belief_updater.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "planning/belief_road_map.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {

// Evaluate using expected determinant
struct ExpectedDeterminant {
    bool position_only;
};

struct ExpectedTrace {
    bool position_only;
};

// Use the determinant at the given percentile to evaluate uncertainty
struct ValueAtRiskDeterminant {
    double percentile;
};

// Compute the probability mass inside of the region as a measure of uncertainty
struct ProbMassInRegion {
    double position_x_half_width_m;
    double position_y_half_width_m;
    double heading_half_width_rad;
};

using UncertaintySizeOptions =
    std::variant<ExpectedDeterminant, ExpectedTrace, ValueAtRiskDeterminant, ProbMassInRegion>;

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
    UncertaintySizeOptions uncertainty_size_options;
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

    UncertaintySizeOptions uncertainty_size_options;
    std::optional<SampledBeliefOptions> sampled_belief_options;
    std::optional<time::RobotTimestamp::duration> timeout;
};

struct PathConstrainedBeliefPlanResult {
    std::vector<int> plan;
    Eigen::Matrix3d expected_cov;
};

struct ExpectedBeliefPlanResult {
    std::vector<int> nodes;
    double log_probability_mass_tracked;
};

struct ExpectedBeliefRoadMapOptions {
    int num_configuration_samples;
    int seed;
    std::optional<time::RobotTimestamp::duration> timeout;
    BeliefRoadMapOptions brm_options;
};

RobotBelief evaluate_path(const std::vector<int> &path, const RobotBelief &initial_belief,
                              const planning::BeliefUpdater<RobotBelief> &updater);

std::vector<RobotBelief> evaluate_paths_with_configuration(
    const std::vector<std::vector<int>> &paths, const EkfSlam &ekf,
    const planning::RoadMap &road_map, const double max_sensor_range_m,
    const std::vector<int> &present_beacons);

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

template <typename T>
std::function<double(const T &)> make_uncertainty_size(const UncertaintySizeOptions &);

template <>
std::function<double(const RobotBelief &)> make_uncertainty_size(const UncertaintySizeOptions &);
template <>
std::function<double(const LandmarkRobotBelief &)> make_uncertainty_size(
    const UncertaintySizeOptions &);

}  // namespace robot::experimental::beacon_sim
