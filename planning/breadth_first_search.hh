
#pragma once

#include <algorithm>
#include <deque>
#include <functional>
#include <optional>
#include <vector>

namespace robot::planning {
template <typename State>
struct Node {
    State state;
    std::optional<int> maybe_parent_idx;
    double cost;
    bool should_skip;
};

template <typename State>
struct Successor {
    State state;
    double edge_cost;
};

template <typename State>
struct BreadthFirstResult {
    std::vector<State> path;
    int num_nodes_expanded;
    int num_nodes_visited;
};

enum class ShouldQueueResult {
    SKIP,
    QUEUE,
    QUEUE_AND_CLEAR_MATCHING_IN_OPEN,
};

template <typename State>
using SuccessorFunc = std::function<std::vector<Successor<State>>(const State &start)>;

template <typename State>
using ShouldQueueFunc = std::function<ShouldQueueResult(
    const Successor<State> &, const int parent_idx, const std::vector<Node<State>> &node_list)>;

template <typename State>
using GoalCheckFunc = std::function<bool(const Node<State> &)>;

template <typename State>
using IdentifyPathEndFunc = std::function<std::optional<int>(const std::vector<Node<State>> &)>;

template <typename State>
std::optional<BreadthFirstResult<State>> breadth_first_search(
    const State &initial_state, const SuccessorFunc<State> &successors_for_state,
    const ShouldQueueFunc<State> &should_queue_check, const GoalCheckFunc<State> &goal_check_func,
    const IdentifyPathEndFunc<State> &identify_end_func) {
    int nodes_expanded = 0;
    int nodes_visited = 0;

    std::vector<Node<State>> nodes = {
        {.state = initial_state, .maybe_parent_idx = {}, .cost = 0.0, .should_skip = false}};
    std::deque<int> node_idx_queue = {0};
    while (!node_idx_queue.empty()) {
        // Pop the front of the queue
        const int node_idx = node_idx_queue.front();
        node_idx_queue.pop_front();
        // Make a copy to avoid invalidated references when pushing back on nodes
        const Node<State> n = nodes.at(node_idx);
        nodes_expanded++;
        if (n.should_skip) {
            continue;
        }
        if (goal_check_func(n)) {
            break;
        }

        // For each neighbor in the queue
        for (const auto &successor : successors_for_state(n.state)) {
            nodes_visited++;

            // Check if we should add this node to the queue
            const auto should_queue = should_queue_check(successor, node_idx, nodes);

            if (should_queue == ShouldQueueResult::SKIP) {
                continue;
            }

            nodes.push_back(Node<State>{.state = successor.state,
                                        .maybe_parent_idx = node_idx,
                                        .cost = n.cost + successor.edge_cost,
                                        .should_skip = false});

            if (should_queue == ShouldQueueResult::QUEUE_AND_CLEAR_MATCHING_IN_OPEN) {
                for (int node_idx : node_idx_queue) {
                    if (nodes.at(node_idx).state == nodes.back().state) {
                        nodes.at(node_idx).should_skip = true;
                    }
                }
            }
            node_idx_queue.push_back(nodes.size() - 1);
        }
    }

    std::optional<int> end_idx = identify_end_func(nodes);
    std::vector<State> path;
    while (end_idx.has_value()) {
        path.push_back(nodes.at(end_idx.value()).state);
        end_idx = nodes.at(end_idx.value()).maybe_parent_idx;
    }

    std::reverse(path.begin(), path.end());

    return BreadthFirstResult<State>{
        .path = path,
        .num_nodes_expanded = nodes_expanded,
        .num_nodes_visited = nodes_visited,
    };
}
}  // namespace robot::planning
