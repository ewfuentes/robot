
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
    double info_lower_bound;
    double cost_to_go;
    std::vector<int> path_to_goal;
};

InformationLowerBoundResult information_lower_bound_search(
    const planning::RoadMap &road_map, const int start_idx, const int end_idx,
    const double start_information, const double end_information_lower_bound,
    const LowerBoundReversePropagator &propagator);

namespace detail {
struct InProgressPath {
    double info_lower_bound;
    double cost_to_go;
    std::vector<int> path_to_goal;

    bool operator>(const InProgressPath &other) const { return cost_to_go > other.cost_to_go; }
};

struct MergeResult {
    bool should_merge;
    std::vector<int> to_boot;
};

MergeResult should_merge(const std::vector<InProgressPath> &existing,
                         const InProgressPath &new_path);
}  // namespace detail
}  // namespace robot::experimental::beacon_sim
