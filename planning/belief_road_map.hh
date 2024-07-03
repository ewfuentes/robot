
#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <variant>

#include "Eigen/Dense"
#include "planning/breadth_first_search.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::planning {
template <typename Belief>
struct BRMPlan {
    std::vector<int> nodes;
    std::vector<Belief> beliefs;
};

template <typename Belief>
using BeliefUpdater =
    std::function<Belief(const Belief &initial_belief, const int start_idx, const int end_idx)>;

struct NoBacktrackingOptions {};
struct MinUncertaintyToleranceOptions {
    double uncertainty_tolerance;
};

using BRMSearchOptions = std::variant<NoBacktrackingOptions, MinUncertaintyToleranceOptions>;

namespace detail {
template <typename Belief>
struct BRMSearchState {
    Belief belief;
    int node_idx;
    bool operator==(const BRMSearchState &other) const {
        return (node_idx == other.node_idx) && (belief == other.belief);
    }
};

template <typename Belief, typename UncertaintySize>
std::vector<BFSSuccessor<BRMSearchState<Belief>>> successors_for_state(
    const BRMSearchState<Belief> &state, const RoadMap &road_map,
    const BeliefUpdater<Belief> &belief_updater, const UncertaintySize &uncertainty_size) {

    std::vector<BFSSuccessor<BRMSearchState<Belief>>> out;
    for (const auto &[other_node_id, other_node_in_local] : road_map.neighbors(state.node_idx)) {
        const Belief new_belief = belief_updater(state.belief, state.node_idx, other_node_id);
        out.push_back(BFSSuccessor<BRMSearchState<Belief>>{
            .state =
                BRMSearchState<Belief>{
                    .belief = new_belief,
                    .node_idx = other_node_id,
                },
            .edge_cost = uncertainty_size(new_belief) - uncertainty_size(state.belief),
        });
    }
    return out;
}

template <typename Belief>
BRMPlan<Belief> brm_plan_from_bfs_result(
    const BreadthFirstResult<BRMSearchState<Belief>> &bfs_result) {
    BRMPlan<Belief> out;
    for (const auto &plan_node : bfs_result.path) {
        out.nodes.push_back(plan_node.node_idx);
        out.beliefs.push_back(plan_node.belief);
    }
    return out;
}
}  // namespace detail

using ShouldTerminateCallback = std::function<bool()>;

// Belief is a type that represents the mean and uncertainty of our current state.
// It is expected that the following methods are defined:
// A std::hash<Belief> specialization should be available
// bool operator==(const Belief &a, const Belief &b) computes if the beliefs are equal
template <typename Belief, typename UncertaintySize>
std::optional<BRMPlan<Belief>> plan(
    const RoadMap &road_map, const Belief &initial_belief,
    const BeliefUpdater<Belief> &belief_updater, const UncertaintySize &uncertainty_size,
    const BRMSearchOptions &options,
    ShouldTerminateCallback should_terminate_callback = []() { return false; }) {
    using SearchState = detail::BRMSearchState<Belief>;
    // Find nearest node to start and end states
    const SuccessorFunc<SearchState> successors_func =
        [&belief_updater, &road_map, &uncertainty_size](const SearchState &state) {
            return detail::successors_for_state(state, road_map, belief_updater, uncertainty_size);
        };

    const GoalCheckFunc<SearchState> goal_check_func =
        [&should_terminate_callback](const Node<SearchState> &) {
            return should_terminate_callback();
        };

    const IdentifyPathEndFunc<SearchState> identify_end_func =
        [&uncertainty_size](const std::vector<Node<SearchState>> &nodes) -> std::optional<int> {
        const auto iter = std::min_element(
            nodes.begin(), nodes.end(),
            [&uncertainty_size](const Node<SearchState> &a, const Node<SearchState> &b) {
                const bool is_a_goal_node = a.state.node_idx == RoadMap::GOAL_IDX;
                const bool is_b_goal_node = b.state.node_idx == RoadMap::GOAL_IDX;

                if (is_a_goal_node && is_b_goal_node) {
                    return uncertainty_size(a.state.belief) < uncertainty_size(b.state.belief);
                } else if (is_a_goal_node) {
                    return true;
                } else {
                    return false;
                }
            });
        if (iter == nodes.end() || iter->state.node_idx != RoadMap::GOAL_IDX) {
            return std::nullopt;
        }
        return std::distance(nodes.begin(), iter);
    };

    std::unordered_map<int, double> min_uncertainty_from_node;
    auto should_queue_check = [&]() -> ShouldQueueFunc<SearchState> {
        if (std::holds_alternative<MinUncertaintyToleranceOptions>(options)) {
            const auto &min_uncertainty_opts = std::get<MinUncertaintyToleranceOptions>(options);
            return [uncertainty_tolerance = min_uncertainty_opts.uncertainty_tolerance,
                    &min_uncertainty_from_node, &uncertainty_size](
                       const BFSSuccessor<SearchState> &successor, const int parent_idx,
                       const std::vector<Node<SearchState>> &nodes) -> ShouldQueueResult {
                const int node_idx = successor.state.node_idx;
                const double uncertainty = uncertainty_size(successor.state.belief);
                const auto iter = min_uncertainty_from_node.find(node_idx);

                if (iter != min_uncertainty_from_node.end()) {
                    const bool should_queue =
                        iter->second * (1 - uncertainty_tolerance) > uncertainty;
                    if (should_queue) {
                        min_uncertainty_from_node.at(node_idx) = uncertainty;
                    }
                    return should_queue ? ShouldQueueResult::QUEUE : ShouldQueueResult::SKIP;
                }
                std::optional<int> prev_idx = parent_idx;
                while (prev_idx.has_value()) {
                    if (nodes.at(prev_idx.value()).state == successor.state) {
                        break;
                    }
                    prev_idx = nodes.at(prev_idx.value()).maybe_parent_idx;
                }

                if (!prev_idx.has_value()) {
                    // If we haven't seen this node before on our path, allow it to be added
                    min_uncertainty_from_node[node_idx] = uncertainty_size(successor.state.belief);
                    return ShouldQueueResult::QUEUE;
                } else if (uncertainty_size(successor.state.belief) <
                           ((1 - uncertainty_tolerance) *
                            uncertainty_size(nodes[prev_idx.value()].state.belief))) {
                    // We're revisiting a node, but it has lower uncertainty
                    min_uncertainty_from_node[node_idx] = uncertainty_size(successor.state.belief);
                    return ShouldQueueResult::QUEUE;
                }
                // We've seen this node before, but it's not much better than last time
                return ShouldQueueResult::SKIP;
            };
        } else {
            return [&min_uncertainty_from_node, &uncertainty_size](
                       const BFSSuccessor<SearchState> &successor, const int parent_idx,
                       const std::vector<Node<SearchState>> &nodes) -> ShouldQueueResult {
                // If node was previously on path, skip it
                std::optional<int> path_node_idx = parent_idx;
                while (path_node_idx.has_value()) {
                    const auto &path_node = nodes.at(path_node_idx.value());
                    if (path_node.state == successor.state) {
                        return ShouldQueueResult::SKIP;
                    }
                    path_node_idx = path_node.maybe_parent_idx;
                }

                // If the node has previously been visited with a lower uncertainty, don't queue
                // this node
                const auto maybe_prev_uncertainty =
                    min_uncertainty_from_node.find(successor.state.node_idx);
                const double successor_uncertainty_size = uncertainty_size(successor.state.belief);
                const bool should_queue =
                    (maybe_prev_uncertainty == min_uncertainty_from_node.end()) ||
                    successor_uncertainty_size < maybe_prev_uncertainty->second;

                if (should_queue) {
                    min_uncertainty_from_node[successor.state.node_idx] =
                        successor_uncertainty_size;
                    return ShouldQueueResult::QUEUE_AND_CLEAR_MATCHING_IN_OPEN;
                }
                return ShouldQueueResult::SKIP;
            };
        }
    }();

    const auto bfs_result = breadth_first_search(
        detail::BRMSearchState<Belief>{.belief = initial_belief, .node_idx = RoadMap::START_IDX},
        successors_func, should_queue_check, goal_check_func, identify_end_func);

    return bfs_result.has_value()
               ? std::make_optional(detail::brm_plan_from_bfs_result(bfs_result.value()))
               : std::nullopt;
    return std::nullopt;
}
}  // namespace robot::planning

namespace std {
template <typename Belief>
struct hash<robot::planning::detail::BRMSearchState<Belief>> {
    size_t operator()(const robot::planning::detail::BRMSearchState<Belief> &state) const {
        std::hash<Belief> belief_hasher;
        std::hash<int> int_hasher;
        return belief_hasher(state.belief) ^ (int_hasher(state.node_idx) << 3);
    }  // namespace std
};
}  // namespace std
