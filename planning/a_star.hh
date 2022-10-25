
#pragma once

#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace robot::planning {

template <typename State>
struct AStarResult {
    std::vector<State> states;
    double cost;
    int num_nodes_expanded;
    int num_nodes_visited;
};

template <typename State>
struct Successor {
    State state;
    double edge_cost;
};

// Find a path through a graph.
// SuccessorFunc returns a list of successors and must have the interface:
//     Iterable<Successor<State>>(const State &)
// HeuristicFunc returns the estimated cost from the argument node to a goal node. It should have
//     the interface: double(const State &)
// GoalCheck returns true if the node is a goal state.
template <typename State, typename SuccessorFunc, typename HeuristicFunc, typename GoalCheck>
std::optional<AStarResult<State>> a_star(const State &initial_state,
                                         const SuccessorFunc &successors_for_state,
                                         const HeuristicFunc &heuristic,
                                         const GoalCheck &termination_check) {
    struct Node {
        State state;
        std::optional<int> maybe_parent_idx;
        double cost_to_come;
        double est_cost_to_go;
        bool in_open;
        bool should_skip;
    };
    struct Compare {
        std::vector<Node> *nodes;
        bool operator()(const int a, const int b) const {
            // This operator should return true if a should be expanded after b;
            const auto &head_a = nodes->at(a);
            const auto &head_b = nodes->at(b);
            const double cost_a = head_a.cost_to_come + head_a.est_cost_to_go;
            const double cost_b = head_b.cost_to_come + head_b.est_cost_to_go;

            if (cost_a == cost_b) {
                // If the total estimated costs are equal, prefer the one that has been expanded
                // more.
                return head_a.cost_to_come < head_b.cost_to_come;
            }
            return cost_a > cost_b;
        }
    };

    const auto extract_path = [](const int end_idx, const auto &nodes) {
        std::vector<State> out;
        for (std::optional<int> node_idx = end_idx; node_idx.has_value();
             node_idx = nodes.at(node_idx.value()).maybe_parent_idx) {
            const auto &node = nodes.at(node_idx.value());
            out.push_back(node.state);
        }
        std::reverse(out.begin(), out.end());
        return out;
    };

    std::vector<Node> nodes;
    nodes.push_back(Node{
        .state = initial_state,
        .maybe_parent_idx = std::nullopt,
        .cost_to_come = 0,
        .est_cost_to_go = 0,
        .in_open = true,
        .should_skip = false,
    });

    std::priority_queue<int, std::vector<int>, Compare> queue(Compare{.nodes = &nodes});
    queue.push(0);
    std::unordered_map<State, double> expanded_nodes;
    int nodes_expanded = 0;
    while (!queue.empty()) {
        const int node_idx = queue.top();
        auto &curr_node = nodes.at(node_idx);
        curr_node.in_open = false;
        queue.pop();
        if (curr_node.should_skip) {
            continue;
        }

        nodes_expanded++;
        expanded_nodes[curr_node.state] = curr_node.cost_to_come;

        // Goal Check
        if (termination_check(curr_node.state)) {
            // Extract the path
            return AStarResult<State>{
                .states = extract_path(node_idx, nodes),
                .cost = curr_node.cost_to_come,
                .num_nodes_expanded = nodes_expanded,
                .num_nodes_visited = static_cast<int>(nodes.size()),
            };
        }

        // nodes.push_back() can re-alloc, so store a copy of the edge cost
        const double curr_node_cost_to_come = curr_node.cost_to_come;
        for (const Successor<State> &successor : successors_for_state(curr_node.state)) {
            const Node successor_node = Node{
                .state = successor.state,
                .maybe_parent_idx = node_idx,
                .cost_to_come = curr_node_cost_to_come + successor.edge_cost,
                .est_cost_to_go = heuristic(successor.state),
                .in_open = true,
            };
            auto in_expanded_iter = expanded_nodes.find(successor_node.state);
            if (in_expanded_iter != expanded_nodes.end()) {
                if (in_expanded_iter->second <= successor_node.cost_to_come) {
                    continue;
                } else {
                    // Remove the existing element from the closed list
                    expanded_nodes.erase(in_expanded_iter);
                }
            }

            // if the node is in the open set and it's not better than the previous item
            // skip it
            auto in_open_iter =
                std::find_if(nodes.begin(), nodes.end(), [&successor_node](const Node &node) {
                    return node.in_open && node.state == successor_node.state;
                });

            if (in_open_iter != nodes.end()) {
                if (in_open_iter->cost_to_come <= successor_node.cost_to_come) {
                    continue;
                } else {
                    // The new successor is better than the existing item in the queue. We should
                    // skip the existing item.
                    in_open_iter->should_skip = true;
                }
            }

            nodes.push_back(successor_node);
            queue.push(nodes.size() - 1);
        }
    }
    return std::nullopt;
}

}  // namespace robot::planning
