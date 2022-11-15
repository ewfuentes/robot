
#pragma once

#include <functional>
#include <optional>

#include "Eigen/Dense"
#include "planning/a_star.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::planning {

// Interface
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
                                    const Eigen::Vector2d &goal_state);

// Implementation details
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
int find_nearest_node_idx(const RoadMap &road_map, const Belief &belief) {
    const auto iter =
        std::min_element(road_map.points.begin(), road_map.points.end(),
                         [&belief](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                             return distance_to(a, belief) < distance_to(b, belief);
                         });
    return std::distance(road_map.points.begin(), iter);
}

int find_nearest_node_idx(const RoadMap &road_map, const Eigen::Vector2d &state) {
    const auto iter =
        std::min_element(road_map.points.begin(), road_map.points.end(),
                         [&state](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                             return (a - state).squaredNorm() < (b - state).squaredNorm();
                         });
    return std::distance(road_map.points.begin(), iter);
}

template <typename Belief>
std::vector<Successor<BRMSearchState<Belief>>> successors_for_state(
    const BRMSearchState<Belief> &state, const RoadMap &road_map,
    const BeliefUpdater<Belief> &belief_updater, const int nearest_to_start_idx,
    const int nearest_to_end_idx) {
    std::vector<Successor<BRMSearchState<Belief>>> out;
    if (state.node_idx == BRMPlan<Belief>::INITIAL_BELIEF_NODE_IDX) {
        // If we're at the start node, add a successor to the nearest node
        const Belief new_belief =
            belief_updater(state.belief, state.node_idx, nearest_to_start_idx);
        out.push_back(Successor<BRMSearchState<Belief>>{
            .state =
                BRMSearchState<Belief>{
                    .belief = new_belief,
                    .node_idx = nearest_to_start_idx,
                },
            .edge_cost = uncertainty_size(new_belief) - uncertainty_size(state.belief),
        });
    } else if (state.node_idx == nearest_to_end_idx) {
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

    // This step cost is added to the edge cost to give preference to shorter paths
    constexpr double STEP_COST = 1e-6;
    for (int i = 0; i < static_cast<int>(road_map.points.size()); i++) {
        if (road_map.adj(state.node_idx, i)) {
            // Queue up each neighbor
            const Belief new_belief = belief_updater(state.belief, state.node_idx, i);
            out.push_back(Successor<BRMSearchState<Belief>>{
                .state =
                    BRMSearchState<Belief>{
                        .belief = new_belief,
                        .node_idx = i,
                    },
                .edge_cost =
                    uncertainty_size(new_belief) - uncertainty_size(state.belief) + STEP_COST,
            });
        }
    }
    return out;
}

template <typename Belief>
BRMPlan<Belief> brm_plan_from_a_star_result(
    const AStarResult<BRMSearchState<Belief>> &a_star_result) {
    BRMPlan<Belief> out;
    for (const auto &plan_node : a_star_result.states) {
        out.nodes.push_back(plan_node.node_idx);
        out.beliefs.push_back(plan_node.belief);
    }
    return out;
}
}  // namespace detail

template <typename Belief>
std::optional<BRMPlan<Belief>> plan(const RoadMap &road_map, const Belief &initial_belief,
                                    const BeliefUpdater<Belief> &belief_updater,
                                    const Eigen::Vector2d &goal_state) {
    // Find nearest node to start and end states
    const int nearest_to_start_idx = detail::find_nearest_node_idx(road_map, initial_belief);
    const int nearest_to_end_idx = detail::find_nearest_node_idx(road_map, goal_state);

    const auto successors_func = [nearest_to_start_idx, nearest_to_end_idx, &belief_updater,
                                  &road_map](const detail::BRMSearchState<Belief> &state) {
        return detail::successors_for_state(state, road_map, belief_updater, nearest_to_start_idx,
                                            nearest_to_end_idx);
    };

    const auto heuristic_func = [](const auto &) { return 0.0; };
    const auto goal_check_func = [](const auto &brm_search_state) {
        return brm_search_state.node_idx == BRMPlan<Belief>::GOAL_BELIEF_NODE_IDX;
    };

    const auto a_star_result =
        a_star(detail::BRMSearchState<Belief>{.belief = initial_belief,
                                              .node_idx = BRMPlan<Belief>::INITIAL_BELIEF_NODE_IDX},
               successors_func, heuristic_func, goal_check_func);

    return a_star_result.has_value()
               ? std::make_optional(detail::brm_plan_from_a_star_result(a_star_result.value()))
               : std::nullopt;
    return std::nullopt;
}
}  // namespace robot::planning

namespace std {
template <typename Belief>
struct std::hash<robot::planning::detail::BRMSearchState<Belief>> {
    size_t operator()(const robot::planning::detail::BRMSearchState<Belief> &state) const {
        std::hash<Belief> belief_hasher;
        std::hash<int> int_hasher;
        return belief_hasher(state.belief) ^ (int_hasher(state.node_idx) << 3);
    }  // namespace std
};
}  // namespace std
