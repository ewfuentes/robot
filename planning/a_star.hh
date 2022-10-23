
#pragma once

#include <optional>
#include <queue>
#include <unordered_set>
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
// if is_consistent_heuristic is true, the algorithm will track which nodes have been expanded.
//      Note that this will only return the optimal path if the heuristic is consistent
template <typename State, typename SuccessorFunc, typename HeuristicFunc, typename GoalCheck>
std::optional<AStarResult<State>> a_star(const State &initial_state,
                                         const SuccessorFunc &successors_for_state,
                                         const HeuristicFunc &heuristic,
                                         const GoalCheck &termination_check) {
    struct PathNode {
        State state;
        std::optional<int> maybe_parent_idx;
        double cost_to_come;
        double est_cost_to_go;
    };

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
    std::unordered_set<State> expanded_nodes;
    int nodes_expanded = 0;
    while (!queue.empty()) {
        nodes_expanded++;
        const int path_node_idx = queue.top();
        queue.pop();
        const auto &curr_node = path_nodes.at(path_node_idx);

        expanded_nodes.insert(curr_node.state);

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

        // For each neighbor
        for (const Successor<State> &successor : successors_for_state(curr_node.state)) {
            // Check if it's already been expanded
            if (expanded_nodes.contains(successor.state)) {
                continue;
            }
            // If not, add it to the priority queue, using the cost to come and the estimated cost
            // to go
            path_nodes.push_back(PathNode{
                .state = successor.state,
                .maybe_parent_idx = path_node_idx,
                .cost_to_come = curr_node.cost_to_come + successor.edge_cost,
                .est_cost_to_go = heuristic(successor.state),
            });
            queue.push(path_nodes.size() - 1);
        }
    }
    return std::nullopt;
}

}  // namespace robot::planning
