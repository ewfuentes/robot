
#pragma once

#include <iostream>
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

template <typename State>
struct Node {
    State state;
    std::optional<int> maybe_parent_idx;
    double cost_to_come;
    double est_cost_to_go;
    bool in_open;
    bool in_closed;
};

template <typename State>
std::ostream &operator<<(std::ostream &out, const Node<State> &node) {
    out << "{S: " << node.state << " f: " << node.cost_to_come + node.est_cost_to_go
        << " g: " << node.cost_to_come << " h: " << node.est_cost_to_go << "}";
    return out;
}

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
    using PathNode = Node<State>;
    struct Compare {
        std::vector<PathNode> &path_nodes;
        bool operator()(const int a, const int b) const {
            const auto &head_a = path_nodes.at(a);
            const auto &head_b = path_nodes.at(b);
            return (head_a.cost_to_come + head_a.est_cost_to_go) >
                   (head_b.cost_to_come + head_b.est_cost_to_go);
        }
    };

    const auto extract_path = [](const int end_idx, const auto &path_nodes) {
        std::vector<State> out;
        for (std::optional<int> node_idx = end_idx; node_idx.has_value();
             node_idx = path_nodes.at(node_idx.value()).maybe_parent_idx) {
            const auto &path_node = path_nodes.at(node_idx.value());
            out.push_back(path_node.state);
        }
        std::reverse(out.begin(), out.end());
        return out;
    };

    std::vector<PathNode> path_nodes;
    path_nodes.push_back(PathNode{
        .state = initial_state,
        .maybe_parent_idx = std::nullopt,
        .cost_to_come = 0,
        .est_cost_to_go = 0,
    });

    std::priority_queue<int, std::vector<int>, Compare> queue(Compare{.path_nodes = path_nodes});
    queue.push(0);
    std::unordered_map<State, double> expanded_nodes;
    int nodes_expanded = 0;
    while (!queue.empty()) {
        nodes_expanded++;
        const int path_node_idx = queue.top();
        queue.pop();
        auto &curr_node = path_nodes.at(path_node_idx);
        std::cout << "Popping " << curr_node << std::endl;

        expanded_nodes[curr_node.state] = curr_node.cost_to_come;
        curr_node.in_closed = true;

        // Goal Check
        if (termination_check(curr_node.state)) {
            // Extract the path
            return AStarResult<State>{
                .states = extract_path(path_node_idx, path_nodes),
                .cost = curr_node.cost_to_come,
                .num_nodes_expanded = nodes_expanded,
                .num_nodes_visited = static_cast<int>(path_nodes.size()),
            };
        }

        // push_back can re-alloc, so store a copy of the edge cost
        const double curr_node_cost_to_come = curr_node.cost_to_come;
        for (const Successor<State> &successor : successors_for_state(curr_node.state)) {
            const PathNode successor_node = PathNode{
                .state = successor.state,
                .maybe_parent_idx = path_node_idx,
                .cost_to_come = curr_node_cost_to_come + successor.edge_cost,
                .est_cost_to_go = heuristic(successor.state),
                .in_open = false,
                .in_closed = false,
            };
            // if the node has been expanded and it's worse than the previous expansion
            // skip it
            const auto expanded_iter = expanded_nodes.find(successor_node.state);
            if (expanded_iter != expanded_nodes.end() &&
                expanded_iter->second <= successor_node.cost_to_come) {
                std::cout << "\tAlready expanded " << successor_node.state << std::endl;
                continue;
            }
            // if the node is in the open set and it's worse than the previous expansion
            // skip it


            path_nodes.push_back(successor_node);
            std::cout << "\tQueuing " << path_nodes.back() << std::endl;
            queue.push(path_nodes.size() - 1);
            path_nodes.back().in_open = true;
        }
    }
    return std::nullopt;
}

}  // namespace robot::planning
