
#pragma once

#include <optional>
#include <queue>
#include <unordered_set>
#include <vector>

namespace robot::planning {

template <typename Node>
struct AStarResult {
    std::vector<Node> nodes;
    double cost;
    int num_nodes_expanded;
    int num_nodes_visited;
};

template <typename Node>
struct Successor {
    Node node;
    double edge_cost;
};

// Find a path through a graph.
// SuccessorFunc returns a list of successors and must have the interface:
//     Iterable<Successor<Node>>(const Node &)
// HeuristicFunc returns the estimated cost from the argument node to a goal node. It should have
//     the interface: double(const Node &)
// GoalCheck returns true if the node is a goal state.
// if is_consistent_heuristic is true, the algorithm will track which nodes have been expanded.
//      Note that this will only return the optimal path if the heuristic is consistent
template <typename Node, typename SuccessorFunc, typename HeuristicFunc, typename GoalCheck>
std::optional<AStarResult<Node>> a_star(const Node &start_node,
                                        const SuccessorFunc &successors_for_node,
                                        const HeuristicFunc &heuristic,
                                        const GoalCheck &termination_check,
                                        const bool is_consistent_heuristic = false) {
    struct PathNode {
        Node node;
        std::optional<int> maybe_parent_idx;
        double cost_to_come;
        double est_cost_to_go;
    };

    std::vector<PathNode> path_nodes;
    path_nodes.push_back(PathNode{
        .node = start_node,
        .maybe_parent_idx = std::nullopt,
        .cost_to_come = 0,
        .est_cost_to_go = 0,
    });

    struct Compare {
        std::vector<PathNode> &path_nodes;
        bool operator()(const int a, const int b) {
            const auto &head_a = path_nodes.at(a);
            const auto &head_b = path_nodes.at(b);
            return (head_a.cost_to_come + head_a.est_cost_to_go) >
                   (head_b.cost_to_come + head_b.est_cost_to_go);
        }
    };

    std::priority_queue<int, std::vector<int>, Compare> queue(Compare{.path_nodes = path_nodes});
    queue.push(0);

    std::unordered_set<Node> expanded_nodes;

    int nodes_expanded = 0;
    const auto extract_path = [](const int end_idx, const auto &path_nodes) {
        std::vector<Node> out;
        int node_idx = end_idx;
        while (true) {
            const auto &path_node = path_nodes.at(node_idx);
            out.push_back(path_node.node);
            if (path_node.maybe_parent_idx.has_value()) {
                node_idx = path_node.maybe_parent_idx.value();
            } else {
                break;
            }
        }
        std::reverse(out.begin(), out.end());
        return out;
    };

    while (!queue.empty()) {
        nodes_expanded++;
        const int path_node_idx = queue.top();
        queue.pop();
        const auto &curr_node = path_nodes.at(path_node_idx);

        if (is_consistent_heuristic) {
            expanded_nodes.insert(curr_node.node);
        }

        // Goal Check
        if (termination_check(curr_node.node)) {
            // Extract the path
            return AStarResult<Node>{
                .nodes = extract_path(path_node_idx, path_nodes),
                .cost = curr_node.cost_to_come,
                .num_nodes_expanded = nodes_expanded,
                .num_nodes_visited = static_cast<int>(path_nodes.size()),
            };
        }

        for (const Successor<Node> &successor : successors_for_node(curr_node.node)) {
            if (!expanded_nodes.empty() && expanded_nodes.contains(successor.node)) {
                continue;
            }
            path_nodes.push_back(PathNode{
                .node = successor.node,
                .maybe_parent_idx = path_node_idx,
                .cost_to_come = curr_node.cost_to_come + successor.edge_cost,
                .est_cost_to_go = heuristic(successor.node),
            });
            queue.push(path_nodes.size() - 1);
        }
    }
    return std::nullopt;
}

}  // namespace robot::planning
