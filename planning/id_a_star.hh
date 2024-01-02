
#pragma once

#include <algorithm>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

namespace robot::planning {

template <typename State>
struct IDAStarResult {
    std::vector<State> states;
    double cost;
    int num_nodes_visited;
};

template <typename State>
struct Successor {
    State state;
    double edge_cost;
};

namespace detail {
struct Failure {
    double next_cost_bound;
};

template <typename State>
struct Success {
    std::vector<State> states;
    double cost;
};

template <typename State>
struct CostLimitedDFSResult {
    int num_nodes_visited;
    std::variant<Failure, Success<State>> result;
};

template <typename State, typename SuccessorFunc, typename HeuristicFunc, typename GoalCheck>
CostLimitedDFSResult<State> cost_limited_dfs(const State &initial_state,
                                             const SuccessorFunc &successors_for_state,
                                             const HeuristicFunc &heuristic,
                                             const GoalCheck &termination_check,
                                             const double cost_bound);
}  // namespace detail

// Implementation of Iterative Deepening A* (IDA*) search.
// SuccessorFunc returns a list of successors and must have the interface:
//     Iterable<Successor<State>>(const State &)
// HeuristicFunc returns the estimated cost from the argument node to a goal node. It should have
//     the interface: double(const State &)
// GoalCheck returns true if the node is a goal state.
template <typename State, typename SuccessorFunc, typename HeuristicFunc, typename GoalCheck>
std::optional<IDAStarResult<State>> id_a_star(const State &initial_state,
                                              const SuccessorFunc &successors_for_state,
                                              const HeuristicFunc &heuristic,
                                              const GoalCheck &termination_check) {
    bool run = true;
    double cost_bound = heuristic(initial_state);
    int num_nodes_visited = 0;
    while (run) {
        // Run a depth-first search with a cost bound.
        const auto result = detail::cost_limited_dfs(initial_state, successors_for_state, heuristic,
                                                     termination_check, cost_bound);
        num_nodes_visited += result.num_nodes_visited;

        if (std::holds_alternative<detail::Failure>(result.result)) {
            // We have not found a solution, so we increase the cost bound and try again.
            cost_bound = std::get<detail::Failure>(result.result).next_cost_bound;
            continue;
        }

        // We have found a solution.
        const auto &success = std::get<detail::Success<State>>(result.result);
        return IDAStarResult<State>{
            .states = std::move(success.states),
            .cost = success.cost,
            .num_nodes_visited = num_nodes_visited,
        };
    }

    return std::nullopt;
}

namespace detail {
template <typename State, typename SuccessorFunc, typename HeuristicFunc, typename GoalCheck>
CostLimitedDFSResult<State> cost_limited_dfs(const State &initial_state,
                                             const SuccessorFunc &successors_for_state,
                                             const HeuristicFunc &heuristic,
                                             const GoalCheck &termination_check,
                                             const double cost_bound) {
    struct StackFrame {
        State state;
        double cost_to_come;
        double est_cost_to_go;
        std::optional<std::vector<Successor<State>>> successors;
    };

    std::vector<StackFrame> stack = {{.state = initial_state,
                                      .cost_to_come = 0,
                                      .est_cost_to_go = heuristic(initial_state),
                                      .successors = std::nullopt}};

    int num_nodes_visited = 0;
    double next_cost_bound = std::numeric_limits<double>::infinity();
    while (!stack.empty()) {
        auto &frame = stack.back();

        if (!frame.successors.has_value()) {
            // We have not yet computed the successors for this node.
            if (termination_check(frame.state)) {
                // We have found a solution.
                std::vector<State> states;
                std::transform(stack.begin(), stack.end(), std::back_inserter(states),
                               [](const auto &frame) { return frame.state; });
                return CostLimitedDFSResult<State>{
                    .num_nodes_visited = num_nodes_visited,
                    .result =
                        Success<State>{
                            .states = std::move(states),
                            .cost = frame.cost_to_come,
                        },
                };
            }
            frame.successors = successors_for_state(frame.state);
        }

        if (frame.successors->empty()) {
            // We have no more successors to expand from this node.
            stack.pop_back();
            continue;
        }

        // We have successors to expand from this node.
        const auto next_successor = frame.successors->back();
        frame.successors->pop_back();
        const double next_cost_to_come = frame.cost_to_come + next_successor.edge_cost;
        const double next_est_cost_to_go = heuristic(next_successor.state);
        const double next_cost = next_cost_to_come + next_est_cost_to_go;
        if (next_cost > cost_bound) {
            // This successor is too expensive, so we skip it.
            next_cost_bound = std::min(next_cost_bound, next_cost);
            continue;
        }

        for (const auto &other_frame : stack) {
            if (other_frame.state == next_successor.state) {
                // We have already expanded this node, so we skip it.
                continue;
            }
        }

        num_nodes_visited += 1;
        // We have not yet found a solution, so we push the successor onto the stack.
        stack.push_back(StackFrame{
            .state = std::move(next_successor.state),
            .cost_to_come = next_cost_to_come,
            .est_cost_to_go = next_est_cost_to_go,
            .successors = std::nullopt,
        });
    }
    return CostLimitedDFSResult<State>{
        .num_nodes_visited = num_nodes_visited,
        .result =
            Failure{
                .next_cost_bound = next_cost_bound,
            },
    };
}
}  // namespace detail
}  // namespace robot::planning