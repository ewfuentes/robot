
#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <optional>

#include "Eigen/Dense"
#include "planning/breadth_first_search.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::planning {
template <typename Belief>
struct BRMPlan {
    static constexpr int INITIAL_BELIEF_NODE_IDX = -1;
    static constexpr int GOAL_BELIEF_NODE_IDX = -2;

    std::vector<int> nodes;
    std::vector<Belief> beliefs;
};

template <typename Belief>
using BeliefUpdater =
    std::function<Belief(const Belief &initial_belief, const int start_idx, const int end_idx)>;

// Belief is a type that represents the mean and uncertainty of our current state.
// It is expected that the following methods are defined:
// - double distance_to(const Eigen::Vector2d &pt, const Belief &belief)
//      This computes the expected distance between a roadmap point and the belief.
//      This is used to to connect the initial belief to the road map
// - double uncertainty_size(const Belief &belief)
//      This computes a measure of how large the uncertainty of the belief is. Common choices are
//      the determinant or the trace of the covariance matrix.
// A std::hash<Belief> specialization should be available
// bool operator==(const Belief &a, const Belief &b) computes if the beliefs are equal
template <typename Belief>
std::optional<BRMPlan<Belief>> plan(const RoadMap &road_map, const Belief &initial_belief,
                                    const BeliefUpdater<Belief> &belief_updater,
                                    const Eigen::Vector2d &goal_state,
                                    const int num_start_connections,
                                    const int num_goal_connections);

// Implementation details follow from here
namespace detail {
template <typename Belief>
struct BRMSearchState {
    Belief belief;
    int node_idx;
    bool operator==(const BRMSearchState &other) const {
        return (node_idx == other.node_idx) && (belief == other.belief);
    }
};

template <typename Belief>
std::vector<int> find_nearest_node_idxs(const RoadMap &road_map, const Belief &belief,
                                        const int num_to_find) {
    std::vector<int> idxs(road_map.points.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(),
              [&belief, &points = road_map.points](const int &a_idx, const int &b_idx) {
                  return distance_to(points.at(a_idx), belief) <
                         distance_to(points.at(b_idx), belief);
              });

    return std::vector<int>(idxs.begin(),
                            idxs.begin() + std::min(num_to_find, static_cast<int>(idxs.size())));
}

std::vector<int> find_nearest_node_idxs(const RoadMap &road_map, const Eigen::Vector2d &state,
                                        const int num_to_find);

template <typename Belief>
std::vector<Successor<BRMSearchState<Belief>>> successors_for_state(
    const BRMSearchState<Belief> &state, const RoadMap &road_map,
    const BeliefUpdater<Belief> &belief_updater, const std::vector<int> &nearest_to_start,
    const std::vector<int> &nearest_to_end) {
    std::vector<Successor<BRMSearchState<Belief>>> out;
    if (state.node_idx == BRMPlan<Belief>::INITIAL_BELIEF_NODE_IDX) {
        // If we're at the start node, add a successor to the nearest node
        for (const int idx : nearest_to_start) {
            const Belief new_belief = belief_updater(state.belief, state.node_idx, idx);
            out.push_back(Successor<BRMSearchState<Belief>>{
                .state =
                    BRMSearchState<Belief>{
                        .belief = new_belief,
                        .node_idx = idx,
                    },
                .edge_cost = uncertainty_size(new_belief) - uncertainty_size(state.belief),
            });
        }
    } else if (std::find(nearest_to_end.begin(), nearest_to_end.end(), state.node_idx) !=
               nearest_to_end.end()) {
        // If we're at the node nearest to the goal pose, add a successor to the goal
        const Belief new_belief =
            belief_updater(state.belief, state.node_idx, BRMPlan<Belief>::GOAL_BELIEF_NODE_IDX);
        out.push_back(Successor<BRMSearchState<Belief>>{
            .state =
                BRMSearchState<Belief>{
                    .belief = new_belief,
                    .node_idx = BRMPlan<Belief>::GOAL_BELIEF_NODE_IDX,
                },
            .edge_cost = uncertainty_size(new_belief) - uncertainty_size(state.belief),
        });
    }
    if (state.node_idx < 0) {
        // We're at the start node or the goal node, we've already added all the successors
        // available
        return out;
    }

    for (int i = 0; i < static_cast<int>(road_map.points.size()); i++) {
        std::cout << "Edge exists " << state.node_idx << "->" << i << "? "
                  << road_map.adj(i, state.node_idx) << " " << road_map.adj(state.node_idx, i)
                  << std::endl;
        if (road_map.adj(state.node_idx, i)) {
            // Queue up each neighbor
            const Belief new_belief = belief_updater(state.belief, state.node_idx, i);
            out.push_back(Successor<BRMSearchState<Belief>>{
                .state =
                    BRMSearchState<Belief>{
                        .belief = new_belief,
                        .node_idx = i,
                    },
                .edge_cost = uncertainty_size(new_belief) - uncertainty_size(state.belief),
            });
        }
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

template <typename Belief>
std::optional<BRMPlan<Belief>> plan(const RoadMap &road_map, const Belief &initial_belief,
                                    const BeliefUpdater<Belief> &belief_updater,
                                    const Eigen::Vector2d &goal_state,
                                    const int num_start_connections, const int num_goal_connections,
                                    const double uncertainty_tolerance) {
    using SearchState = detail::BRMSearchState<Belief>;
    // Find nearest node to start and end states
    const std::vector<int> nearest_to_start_idxs =
        detail::find_nearest_node_idxs(road_map, initial_belief, num_start_connections);
    const std::vector<int> nearest_to_end_idxs =
        detail::find_nearest_node_idxs(road_map, goal_state, num_goal_connections);

    const auto successors_func = [nearest_to_start_idxs, nearest_to_end_idxs, &belief_updater,
                                  &road_map](const SearchState &state) {
        return detail::successors_for_state(state, road_map, belief_updater, nearest_to_start_idxs,
                                            nearest_to_end_idxs);
    };

    const auto goal_check_func = [](const Node<SearchState> &) { return false; };

    const auto identify_end_func =
        [](const std::vector<Node<SearchState>> &nodes) -> std::optional<int> {
        const auto iter = std::min_element(
            nodes.begin(), nodes.end(), [](const Node<SearchState> &a, const Node<SearchState> &b) {
                const bool is_a_goal_node =
                    a.state.node_idx == BRMPlan<Belief>::GOAL_BELIEF_NODE_IDX;
                const bool is_b_goal_node =
                    b.state.node_idx == BRMPlan<Belief>::GOAL_BELIEF_NODE_IDX;

                if (is_a_goal_node && is_b_goal_node) {
                    return uncertainty_size(a.state.belief) < uncertainty_size(b.state.belief);
                } else if (is_a_goal_node) {
                    return true;
                } else {
                    return false;
                }
            });
        if (iter == nodes.end() || iter->state.node_idx != BRMPlan<Belief>::GOAL_BELIEF_NODE_IDX) {
            return std::nullopt;
        }
        return std::distance(nodes.begin(), iter);
    };

    std::unordered_map<int, double> min_uncertainty_from_node;
    auto should_queue_check = [uncertainty_tolerance, &min_uncertainty_from_node](
                                  const Successor<SearchState> &successor, const int parent_idx,
                                  const std::vector<Node<SearchState>> &nodes,
                                  const bool should_print) {
        if (should_print) {
            std::cout << "Checking edge from " << nodes[parent_idx].state.node_idx << " to "
                      << successor.state.node_idx << std::endl;
        }
        const int node_idx = successor.state.node_idx;
        const double uncertainty = uncertainty_size(successor.state.belief);
        const auto iter = min_uncertainty_from_node.find(node_idx);
        if (iter != min_uncertainty_from_node.end()) {
            const bool should_queue = iter->second * (1 - uncertainty_tolerance) > uncertainty;
            if (should_print) {
                std::cout << "found old uncertainty: " << iter->second
                          << " new uncertainty: " << uncertainty << " should queue? "
                          << should_queue << std::endl;
            }
            if (should_queue) {
                min_uncertainty_from_node[node_idx] = uncertainty;
            }
            return should_queue;
        }
        std::optional<int> prev_idx = parent_idx;
        while (prev_idx.has_value()) {
            if (nodes[prev_idx.value()].state == successor.state) {
                break;
            }
            prev_idx = nodes[prev_idx.value()].maybe_parent_idx;
        }

        if (!prev_idx.has_value()) {
            // If we haven't seen this node before on our path, allow it to be added
            min_uncertainty_from_node[node_idx] = uncertainty_size(successor.state.belief);
            if (should_print) {
                std::cout << "Have not seen, caching and adding to queue" << std::endl;
            }
            return true;
        } else if (uncertainty_size(successor.state.belief) <
                   ((1 - uncertainty_tolerance) *
                    uncertainty_size(nodes[prev_idx.value()].state.belief))) {
            // We're revisiting a node, but it has lower uncertainty
            min_uncertainty_from_node[node_idx] = uncertainty_size(successor.state.belief);
            if (should_print) {
                std::cout << "Have seen before and lower uncertainty, updating caching and adding "
                             "to queue"
                          << std::endl;
            }
            return true;
        }
        // We've seen this node before, but it's not much better than last time
        if (should_print) {
            std::cout << "have seen before and not better" << std::endl;
        }
        return false;
    };

    const auto bfs_result = breadth_first_search(
        detail::BRMSearchState<Belief>{.belief = initial_belief,
                                       .node_idx = BRMPlan<Belief>::INITIAL_BELIEF_NODE_IDX},
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
