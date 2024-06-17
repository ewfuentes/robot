#pragma once

#include <algorithm>
#include <deque>
#include <functional>
#include <optional>
#include <vector>
#include <concepts>
#include <queue>
#include <unordered_map>

namespace robot::planning{
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
struct Result {
    std::vector<State> path;
    int num_nodes_expanded;
    int num_nodes_visited;
};

std::optional<Result<State>> search(
    const State &initial_state, const SuccessorFunc<State> &successors_for_state) {}

}  // namespace robot::planning




/* Pseudocode for Dijkstra's algorithm

func Dijkstra(environment, source){
    for each node in environment.nodes{
        dist[v] = infinity
        prev[v] = undef
        add v to Q
    dist[source] = 0
    }
    while Q != empty{
        u = vertex in Q with min dist[u];
        remove u from Q;

        for each neighbor v of u still in Q:
            alt = dist[u] + Graph.edges(u,v);
            if alt < dist[v]:
                dist[v] = alt;
                prev[v] = u;
    }
    return(dist[u],prev[]);
}

*/
