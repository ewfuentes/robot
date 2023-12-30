
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace robot::experimental::polytope {
struct Puzzle {
    using State = std::vector<int>;
    using Move = std::function<State(const State&)>;
    std::string name;
    std::unordered_map<std::string, Move> actions;
};

std::optional<std::vector<std::string>> solve(const Puzzle& puzzle, const Puzzle::State& initial_state,
                       const Puzzle::State& solution_state, const int num_wildcards);
}  // namespace robot::experimental::polytope