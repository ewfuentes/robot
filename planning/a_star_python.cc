
#include "planning/a_star.hh"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace robot::planning {

PYBIND11_MODULE(a_star_python, m) {
    m.doc() = "Generic A* pathfinding for integer-indexed graphs";

    py::class_<AStarResult<int>>(m, "AStarResult")
        .def_readonly("states", &AStarResult<int>::states)
        .def_readonly("cost", &AStarResult<int>::cost)
        .def_readonly("num_nodes_expanded", &AStarResult<int>::num_nodes_expanded)
        .def_readonly("num_nodes_visited", &AStarResult<int>::num_nodes_visited);

    // Generic A* for graphs with integer node indices
    // Args:
    //   neighbors: list[list[int]] - adjacency list for each node
    //   edge_costs: list[list[float]] - parallel to neighbors, cost of each edge
    //   heuristics: numpy array of shape (N,) - precomputed heuristic value for each node
    //   start_idx: starting node index
    //   goal_idx: goal node index
    //   max_expanded: maximum nodes to expand before giving up (0 = unlimited)
    m.def(
        "find_path",
        [](const std::vector<std::vector<int>> &neighbors,
           const std::vector<std::vector<double>> &edge_costs,
           py::array_t<double> heuristics,  // (N,) precomputed heuristic values
           int start_idx, int goal_idx, int max_expanded) -> std::optional<AStarResult<int>> {
            auto h = heuristics.unchecked<1>();
            int nodes_expanded_count = 0;

            // Successors: neighbors with provided edge costs
            auto successors = [&](int state) {
                std::vector<Successor<int>> result;
                const auto &state_neighbors = neighbors[state];
                const auto &state_costs = edge_costs[state];
                for (size_t i = 0; i < state_neighbors.size(); ++i) {
                    result.push_back({state_neighbors[i], state_costs[i]});
                }
                return result;
            };

            // Heuristic: lookup precomputed value
            auto heuristic = [&](int state) { return h(state); };

            // Goal check with expansion limit
            auto goal_check = [&](int state) {
                nodes_expanded_count++;
                if (max_expanded > 0 && nodes_expanded_count > max_expanded) {
                    // Force termination by pretending we reached the goal
                    // The caller should check if the returned path actually reaches the goal
                    return true;
                }
                return state == goal_idx;
            };

            auto result = a_star(start_idx, successors, heuristic, goal_check);

            // If we hit the expansion limit without finding goal, return nullopt
            if (result.has_value() && max_expanded > 0 && nodes_expanded_count > max_expanded) {
                // Check if we actually reached the goal
                if (result->states.empty() || result->states.back() != goal_idx) {
                    return std::nullopt;
                }
            }

            return result;
        },
        py::arg("neighbors"), py::arg("edge_costs"), py::arg("heuristics"), py::arg("start_idx"),
        py::arg("goal_idx"), py::arg("max_expanded") = 0,
        R"pbdoc(
        Find a path using A* algorithm.

        Args:
            neighbors: Adjacency list for each node (list of list of neighbor indices)
            edge_costs: Cost of each edge, parallel to neighbors (list of list of floats)
            heuristics: Precomputed heuristic value for each node (numpy array of shape (N,))
            start_idx: Starting node index
            goal_idx: Goal node index
            max_expanded: Maximum nodes to expand before giving up (0 = unlimited, default)

        Returns:
            AStarResult with states (path), cost, num_nodes_expanded, num_nodes_visited
            or None if no path found or expansion limit reached.
        )pbdoc");
}

}  // namespace robot::planning
