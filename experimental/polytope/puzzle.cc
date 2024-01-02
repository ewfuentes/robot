
#include "experimental/polytope/puzzle.hh"

#include "planning/id_a_star.hh"
#include "common/check.hh"

namespace std {
// Add a std::hash specialization for tuple<string, vector<int>>
template <>
struct hash<std::tuple<std::string, std::vector<int>>> {
    size_t operator()(const std::tuple<std::string, std::vector<int>> &item) const {
        hash<int> int_hasher;
        const auto &[_, state] = item;
        size_t out = 0;
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
}  // namespace

std::optional<std::vector<std::string>> solve(const Puzzle &puzzle,
                                              const Puzzle::State &initial_state,
                                              const Puzzle::State &solution_state,
                                              const int num_wildcards) {
    const auto initial_search_state = std::make_tuple(std::string(""), initial_state);

    std::unordered_map<std::string, std::vector<std::pair<std::string, Puzzle::Move>>>
        valid_moves_from_move;
    valid_moves_from_move[""] = {};
    for (const auto &key_and_move : puzzle.actions) {
        valid_moves_from_move[""].push_back(key_and_move);
        valid_moves_from_move[key_and_move.first] = {};
        const auto inverse_move = compute_inverse_move(key_and_move.first);
        for (const auto &other_key_and_move : puzzle.actions) {
            if (other_key_and_move.first == inverse_move) {
                continue;
            }
            valid_moves_from_move[key_and_move.first].push_back(other_key_and_move);
        }
    }

    const auto successors_for_state = [valid_moves_from_move](const auto &search_state) {
        const auto &[prev_move, prev_state] = search_state;
        std::vector<planning::Successor<std::tuple<std::string, Puzzle::State>>> out;
        for (const auto &[name, action] : valid_moves_from_move.at(prev_move)) {
            out.push_back({
                .state = std::make_tuple(name, action(prev_state)),
                .edge_cost = 1,
            });
        }
        return out;
    };

    const auto heuristic = [&solution_state](const auto &search_state) -> double {
        int error_count = 0;
        const auto &[move, state] = search_state;
        CHECK(state.size() == solution_state.size());
        for (int i = 0; i < static_cast<int>(state.size()); ++i) {
            if (state[i] != solution_state[i]) {
                ++error_count;
            }
        }
        // One move can fix up to 8 errors, so we divide by 8
        return static_cast<int>(error_count / 8.0 + 0.5);
    };

    const auto termination_check = [&solution_state, num_wildcards](const auto &search_state) {
        int error_count = 0;
        const auto &[move, state] = search_state;
        CHECK(state.size() == solution_state.size());
        for (int i = 0; i < static_cast<int>(state.size()); ++i) {
            if (state[i] != solution_state[i]) {
                ++error_count;
            }
        }
        return error_count <= num_wildcards;
    };

    const auto maybe_solution =
        planning::id_a_star(initial_search_state, successors_for_state, heuristic, termination_check);
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
}  // namespace robot::experimental::polytope