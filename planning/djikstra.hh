
#pragma once

#include <concepts>
#include <optional>
#include <queue>
#include <unordered_map>

namespace robot::planning {

template <typename T>
concept State = requires(T t) {
                    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
                };

template <State S>
struct Successor {
    S state;
    double edge_cost;
};

template <State S>
struct DjikstraResult {
    std::unordered_map<S, double> cost_to_go_from_state;
    std::unordered_map<S, S> back_pointer;
};

template <State S, typename SuccessorFunc, typename TerminationCheck>
DjikstraResult<S> djikstra(const S &initial_state, const SuccessorFunc &successor_for_state,
                           const TerminationCheck &termination_check) {
    struct QueueElement {
        S state;
        std::optional<S> parent;
        double cost_to_go;
    };

    const auto compare = [](const QueueElement &a, const QueueElement &b) -> bool {
        return a.cost_to_go > b.cost_to_go;
    };

    std::priority_queue<QueueElement, std::vector<QueueElement>, decltype(compare)> open_list(
        compare);
    std::unordered_map<S, double> cost_to_go_from_state;
    std::unordered_map<S, S> back_pointer;

    open_list.push({.state = initial_state, .parent = std::nullopt, .cost_to_go = 0.0});
    for (; !open_list.empty() && !termination_check(cost_to_go_from_state); open_list.pop()) {
        // Make a copy since the vector may reallocate
        const auto elem = open_list.top();

        if (cost_to_go_from_state.contains(elem.state)) {
            // This node has previously been visited, we ignore it
            continue;
        }

        cost_to_go_from_state[elem.state] = elem.cost_to_go;
        if (elem.parent.has_value()) {
            back_pointer[elem.state] = elem.parent.value();
        }

        for (const auto &successor : successor_for_state(elem.state)) {
            open_list.push({.state = successor.state,
                            .parent = elem.state,
                            .cost_to_go = elem.cost_to_go + successor.edge_cost});
        }
    }
    return {
        .cost_to_go_from_state = std::move(cost_to_go_from_state),
        .back_pointer = std::move(back_pointer),
    };
}
}  // namespace robot::planning
