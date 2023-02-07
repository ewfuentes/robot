
#pragma once

#include <algorithm>
#include <deque>
#include <iostream>
#include <optional>
#include <vector>

namespace robot::experimental::beacon_sim {
extern bool debug;
}

namespace robot::planning {
template <typename State>
struct Node {
    State state;
    std::optional<int> maybe_parent_idx;
    double cost;
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

template <typename State, typename SuccessorFunc, typename ShouldQueueCheckFunc,
          typename GoalCheckFunc, typename IdentifyPathEndFunc>
std::optional<BreadthFirstResult<State>> breadth_first_search(
    const State &initial_state, const SuccessorFunc &successors_for_state,
    const ShouldQueueCheckFunc &should_queue_check, GoalCheckFunc &goal_check_func,
    IdentifyPathEndFunc &identify_end_func) {
    int nodes_expanded = 0;
    int nodes_visited = 0;

    std::vector<Node<State>> nodes = {
        {.state = initial_state, .maybe_parent_idx = {}, .cost = 0.0}};
    std::deque<int> node_idx_queue = {0};
    bool should_print = false;
    while (!node_idx_queue.empty()) {
        if (experimental::beacon_sim::debug == true) {
            experimental::beacon_sim::debug = false;
            should_print = true;
        }
        if (nodes_expanded % 100000 == 0) {
            std::cout << "Nodes expanded: " << nodes_expanded
                      << " Queue size: " << node_idx_queue.size() << " num nodes: " << nodes.size()
                      << std::endl;
        }
        // Pop the front of the queue
        const int node_idx = node_idx_queue.front();
        node_idx_queue.pop_front();
        // Make a copy to avoid invalidated references when pushing back on nodes
        const Node<State> n = nodes[node_idx];
        nodes_expanded++;
        if (goal_check_func(n)) {
            break;
        }
        if (should_print) {
            std::cout << "Popped node idx: " << node_idx << " node id: " << n.state.node_idx
                      << std::endl;
        }

        // For each neighbor in the queue
        for (const auto &successor : successors_for_state(n.state)) {
            nodes_visited++;
            const bool should_queue = should_queue_check(successor, node_idx, nodes, should_print);

            if (should_print) {
              std::cout << "should queue id " << successor.state.node_idx << "? " << should_queue << std::endl;
            }

            // Check if we should add this node to the queue
            if (should_queue) {
                node_idx_queue.push_back(nodes.size());
                nodes.push_back({.state = successor.state,
                                 .maybe_parent_idx = node_idx,
                                 .cost = n.cost + successor.edge_cost});
            }
        }
        should_print = false;
    }

    std::cout << "Done with search, creating path" << std::endl;

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
