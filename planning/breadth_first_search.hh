
#pragma once

#include <algorithm>
#include <deque>
#include <functional>
#include <optional>
#include <vector>
#include <iostream>

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

template <typename State>
struct ShouldQueueResult {
    Node<State> node;
    bool remove_open_queue_nodes;
};

template <typename State>
using ShouldQueueFunc = std::function<std::optional<ShouldQueueResult<State>>(
    const Successor<State> &, const int parent_idx, const std::vector<Node<State>> &node_list)>;

template <typename State, typename SuccessorFunc, typename GoalCheckFunc,
          typename IdentifyPathEndFunc>
std::optional<BreadthFirstResult<State>> breadth_first_search(
    const State &initial_state, const SuccessorFunc &successors_for_state,
    const ShouldQueueFunc<State> &should_queue_check, GoalCheckFunc &goal_check_func,
    IdentifyPathEndFunc &identify_end_func) {
    int nodes_expanded = 0;
    int nodes_visited = 0;

    std::vector<Node<State>> nodes = {
        {.state = initial_state, .maybe_parent_idx = {}, .cost = 0.0}};
    std::deque<int> node_idx_queue = {0};
    while (!node_idx_queue.empty()) {
        // Pop the front of the queue
        const int node_idx = node_idx_queue.front();
        node_idx_queue.pop_front();
        // Make a copy to avoid invalidated references when pushing back on nodes
        const Node<State> n = nodes[node_idx];
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
            const auto maybe_should_queue_result = should_queue_check(successor, node_idx, nodes);
            if (!maybe_should_queue_result.has_value()) {
                continue;
            }

            nodes.push_back(maybe_should_queue_result->node);
            if (maybe_should_queue_result->remove_open_queue_nodes) {
                std::cout << "Clearing nodes matching: " << maybe_should_queue_result->node.state.node_idx << " new node idx: " << nodes.size()-1 << std::endl;
                for (int node_idx : node_idx_queue) {
                    if (nodes.at(node_idx).state == nodes.back().state) {
                        std::cout << nodes.at(node_idx).state.node_idx << " " << nodes.back().state.node_idx << std::endl;
                        nodes.at(node_idx).should_skip = true;
                    }
                }
            }
            node_idx_queue.push_back(nodes.size());

        }
    }

    std::optional<int> end_idx = identify_end_func(nodes);
    std::vector<State> path;
    while (end_idx.has_value()) {
        path.push_back(nodes[end_idx.value()].state);
        end_idx = nodes[end_idx.value()].maybe_parent_idx;
    }

    std::reverse(path.begin(), path.end());

    return BreadthFirstResult<State>{
        .path = path,
        .num_nodes_expanded = nodes_expanded,
        .num_nodes_visited = nodes_visited,
    };
}
}  // namespace robot::planning
