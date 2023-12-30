
#include "experimental/polytope/puzzle.hh"

#include "planning/a_star.hh"

namespace std {
// Add a std::hash specialization for tuple<string, vector<int>>
template <>
struct hash<std::tuple<std::string, std::vector<int>>> {
    size_t operator()(const std::tuple<std::string, std::vector<int>> &item) const {
        hash<std::string> string_hasher;
        hash<int> int_hasher;
        const auto &[name, state] = item;
        size_t out = string_hasher(name);
        for (const auto &item : state) {
            out ^= int_hasher(item) << 1;
        }
        return out;
    }
};
}  // namespace std

namespace robot::experimental::polytope {
namespace {
std::string compute_inverse_move(const std::string &move_name) {
    if (move_name == "") {
        return "";
    } else if (move_name.starts_with("-")) {
        return move_name.substr(1);
    } else {
        return "-" + move_name;
    }
}
}

std::optional<std::vector<std::string>> solve(const Puzzle& puzzle, const Puzzle::State& initial_state,
                       const Puzzle::State& solution_state, const int num_wildcards) {
    const auto initial_search_state = std::make_tuple(std::string(""), initial_state);
    const auto successors_for_state = [&puzzle](const auto& search_state) {
        const auto &[prev_move, prev_state] = search_state;
        const auto inverse_move = compute_inverse_move(prev_move);
        std::vector<planning::Successor<std::tuple<std::string, Puzzle::State>>> out;
        for (const auto& [name, action] : puzzle.actions) {
            if (name == inverse_move) {
                continue;
            }
            out.push_back({
                .state = std::make_tuple(name, action(prev_state)),
                .edge_cost = 1,
            });
        }
        return out;
    };

    const auto heuristic = []([[maybe_unused]] const auto &search_state) {return 0.0;};

    const auto termination_check = [&solution_state, num_wildcards](const auto &search_state) {
        int error_count = 0;
        const auto &[move, state] = search_state;
        for (int i = 0; i < static_cast<int>(state.size()); ++i) {
            if (state[i] != solution_state[i]) {
                ++error_count;
            }
        }
        return error_count <= num_wildcards;
    };

    const auto maybe_solution = planning::a_star(initial_search_state, successors_for_state,
                                       heuristic, termination_check);
    if (!maybe_solution.has_value()) {
        return std::nullopt;
    }
    std::vector<std::string> out;
    for (const auto &search_state : maybe_solution->states) {
        const auto &[move, state] = search_state;
        if (move != "") {
            out.push_back(move);
        }
    }
    return out;
}
}