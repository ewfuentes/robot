
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
struct BFSSuccessor {
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
    SKIP, //init as 0 meaning true or skip
    QUEUE, //init as 1 meaning false of q
    QUEUE_AND_CLEAR_MATCHING_IN_OPEN, //
};

template <typename State>
using SuccessorFunc = std::function<std::vector<BFSSuccessor<State>>(const State &start)>;

template <typename State>
using ShouldQueueFunc = std::function<ShouldQueueResult(
    const BFSSuccessor<State> &, const int parent_idx, const std::vector<Node<State>> &node_list)>;

template <typename State>
using GoalCheckFunc = std::function<bool(const Node<State> &)>;

template <typename State>
using IdentifyPathEndFunc = std::function<std::optional<int>(const std::vector<Node<State>> &)>;

template <typename State>
std::optional<BreadthFirstResult<State>> breadth_first_search(
    const State &initial_state, const SuccessorFunc<State> &successors_for_state,
    const ShouldQueueFunc<State> &should_queue_check, const GoalCheckFunc<State> &goal_check_func,
    const IdentifyPathEndFunc<State> &identify_end_func) {
    int nodes_expanded = 0;//what does this mean?
    int nodes_visited = 0;

    std::vector<Node<State>> nodes = {//initializes vector and puts just init node in it
        {.state = initial_state, .maybe_parent_idx = {}, .cost = 0.0, .should_skip = false}};
    std::deque<int> node_idx_queue = {0};//After initializing first node values, we dequeue it?
    while (!node_idx_queue.empty()) {//While loop. meat of function that runs through BFS while node_idx_queue is not empty
        // Pop the front of the queue
        const int node_idx = node_idx_queue.front();//sets current node index to node at front of queue
        node_idx_queue.pop_front();//removes node_idx current from the front of queue
        // Make a copy to avoid invalidated references when pushing back on nodes
        const Node<State> n = nodes.at(node_idx);//I don't understand
        nodes_expanded++;//Adds one to expanded because we brought a new node to front
        if (n.should_skip) {//as in, goal node is not at this idx then we keep searching
            continue;
        }
        if (goal_check_func(n)) {//checks if n has goal node if so breaks and returns values
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
            node_idx_queue.push_back(nodes.size() - 1);//reduces size because for loops checks one node and reduces size of queue
        }
    }

    std::optional<int> end_idx = identify_end_func(nodes);
    if (!end_idx.has_value()) {
        return std::nullopt;
    }
    std::vector<State> path;
    while (end_idx.has_value()) {
        path.push_back(nodes.at(end_idx.value()).state);
        end_idx = nodes.at(end_idx.value()).maybe_parent_idx;
    }

    std::reverse(path.begin(), path.end());//reverses the vector so that the first value is init state and last is final state

    return BreadthFirstResult<State>{
        .path = path,
        .num_nodes_expanded = nodes_expanded,
        .num_nodes_visited = nodes_visited,
    };
}
}  // namespace robot::planning
