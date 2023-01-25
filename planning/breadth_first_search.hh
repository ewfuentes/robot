
#include <algorithm>
#include <deque>
#include <optional>
#include <vector>

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
    while (!node_idx_queue.empty()) {
        // Pop the front of the queue
        const int node_idx = node_idx_queue.front();
        node_idx_queue.pop_front();
        const Node<State> &n = nodes[node_idx];
        nodes_expanded++;
        if (goal_check_func(n)) {
            break;
        }

        // For each neighbor in the queue
        for (const auto &successor : successors_for_state(n.state)) {
            nodes_visited++;

            // Find the last time we visited this node
            if (should_queue_check(successor, node_idx, nodes)) {
                node_idx_queue.push_back(nodes.size());
                nodes.push_back({.state = successor.state,
                                 .maybe_parent_idx = node_idx,
                                 .cost = n.cost + successor.edge_cost});
            }
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
