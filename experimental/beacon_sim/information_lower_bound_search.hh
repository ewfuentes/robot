
#pragma once

#include <functional>

#include "Eigen/Core"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {

struct PropagationResult {
    double info_lower_bound;
    double edge_cost;
};

// Propagate an information lower bound backwards in time from the end idx to the start idx.
// Note that these indices come from the road map
using LowerBoundReversePropagator = std::function<PropagationResult(
    const int start_idx, const int end_idx, const double lower_bound_at_end)>;

struct InformationLowerBoundResult {
    // A lower bound on the minimum eigenvalue of the information matrix at the goal such that
    // a the information constraint at the goal is satisfied.
    double info_lower_bound;
    // The cost from the start to the goal
    double cost_to_go;
    // The nodes visited by this path
    std::vector<std::tuple<int, double>> path_and_bounds_to_goal;
};

InformationLowerBoundResult information_lower_bound_search(
    const planning::RoadMap &road_map, const double start_information,
    const double end_information_lower_bound, const LowerBoundReversePropagator &propagator);

namespace detail {
struct InProgressPath {
    // A lower bound on the minimum eigenvalue of the information matrix
    double info_lower_bound;
    double cost_to_go;
    // The path traversed starting at the goal and moving back towards the start.
    std::vector<std::tuple<int, double>> path_and_bounds_to_goal;

    bool operator>(const InProgressPath &other) const { return cost_to_go > other.cost_to_go; }
};

struct MergeResult {
    // Whether the new path under consideration should be included in the list of paths to the
    // node in question.
    bool should_merge;
    // The indices of constraints that are dominated by the new path
    std::vector<int> dominated_paths_idxs;
};

MergeResult should_merge(const std::vector<InProgressPath> &existing,
                         const InProgressPath &new_path);
}  // namespace detail
}  // namespace robot::experimental::beacon_sim
