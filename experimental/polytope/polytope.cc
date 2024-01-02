
#include <filesystem>
#include <fstream>
#include <optional>

#include "cxxopts.hpp"
#include "experimental/polytope/puzzle.hh"
#include "nlohmann/json.hpp"
#include "common/check.hh"

using json = nlohmann::json;

namespace robot::experimental::polytope {
namespace {
struct PuzzleSpec {
    int id;
    std::string type;
    std::vector<int> initial_state;
    std::vector<int> solution_state;
    std::vector<std::string> name_from_idx;
    int num_wild_cards;
};

std::vector<std::string> split_on(const std::string& str, const char delim) {
    std::vector<std::string> parts = {};
    auto part_start_iter = str.begin();
    for (auto iter = str.begin(); iter != str.end(); ++iter) {
        if (*iter == delim) {
            parts.push_back(std::string(part_start_iter, iter));
            part_start_iter = iter + 1;
        }
    }
    parts.push_back(std::string(part_start_iter, str.end()));
    return parts;
}

Puzzle parse_puzzle_line(const std::string& name, const std::string& info_string) {
    const auto quote_stripped = info_string.substr(1, info_string.size() - 2);
    const auto double_quoted = std::regex_replace(quote_stripped, std::regex("'"), "\"");
    const auto puzzle_info = json::parse(double_quoted);
    std::unordered_map<std::string, Puzzle::Move> actions;

    for (const auto& [action_name, action_info] : puzzle_info.items()) {
        std::vector<int> permutation(action_info.begin(), action_info.end());

        for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
            const auto iter = std::find(permutation.begin(), permutation.end(), i);
            CHECK(iter != permutation.end());
        }
        // permutation is vector where each element is the index of the element in the new state
        // compute the inverse permutation
        std::vector<int> inverse_permutation(permutation.size());
        for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
            inverse_permutation.at(permutation.at(i)) = i;
        }

        const auto action = [permutation = std::move(permutation)](const Puzzle::State& state) {
            CHECK(state.size() == permutation.size());
            Puzzle::State new_state;
            new_state.reserve(state.size());

            for (int i = 0; i < static_cast<int>(state.size()); ++i) {
                new_state.push_back(state.at(permutation.at(i)));
            }
            return new_state;
        };

        actions[action_name] = action;

        const auto inverse_action = [inverse_permutation = std::move(inverse_permutation)](const Puzzle::State& state) {
            CHECK(state.size() == inverse_permutation.size());
            Puzzle::State new_state;
            new_state.reserve(state.size());
            for (int i = 0; i < static_cast<int>(state.size()); ++i) {
                new_state.push_back(state.at(inverse_permutation.at(i)));
            }
            return new_state;
        };

        actions["-" + action_name] = inverse_action;
    }

    const auto puzzle = Puzzle{
        name,
        actions,
    };
    return puzzle;
}

std::unordered_map<std::string, Puzzle> load_puzzle_info(
    const std::filesystem::path& puzzle_info_file) {
    std::ifstream puzzle_info_stream(puzzle_info_file);

    std::unordered_map<std::string, Puzzle> puzzles;
    for (std::string line; std::getline(puzzle_info_stream, line);) {
        const auto sep = line.find(',');
        const auto name = line.substr(0, sep);
        const auto info_string = line.substr(sep + 1);
        if (name == "puzzle_type") {
            // Skip the header
            continue;
        }
        std::cout << name << std::endl;
        puzzles[name] = parse_puzzle_line(name, info_string);
    }

    return puzzles;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<std::string>> parse_state(
    const std::string& solution_str, const std::string& initial_str) {
    std::vector<std::string> solution_parts = split_on(solution_str, ';');
    std::vector<std::string> initial_parts = split_on(initial_str, ';');

    std::unordered_map<std::string, int> idx_from_name;
    for (const auto& part : solution_parts) {
        if (idx_from_name.count(part) == 0) {
            idx_from_name[part] = idx_from_name.size();
        }
    }

    std::vector<int> solution_state(solution_parts.size());
    std::vector<int> initial_state(initial_parts.size());

    for (int i = 0; i < static_cast<int>(solution_parts.size()); ++i) {
        solution_state.at(i) = idx_from_name.at(solution_parts.at(i));
        initial_state.at(i) = idx_from_name.at(initial_parts.at(i));
    }

    std::vector<std::string> name_from_idx(idx_from_name.size());
    for (const auto& [name, idx] : idx_from_name) {
        name_from_idx.at(idx) = name;
    }

    return std::make_tuple(solution_state, initial_state, name_from_idx);
}

std::optional<PuzzleSpec> parse_puzzle_line(const std::string& line) {
    // Find the position of all commas in the line
    std::vector<std::string> parts = split_on(line, ',');

    if (parts[0] == "id") {
        // Skip the header
        return std::nullopt;
    }

    const auto& [solution_state, initial_state, name_from_idx] = parse_state(parts[2], parts[3]);

    return std::make_optional(PuzzleSpec{
        .id = std::stoi(parts[0]),
        .type = parts[1],
        .initial_state = initial_state,
        .solution_state = solution_state,
        .name_from_idx = name_from_idx,
        .num_wild_cards = std::stoi(parts[4]),
    });
}

std::vector<PuzzleSpec> load_puzzles(const std::filesystem::path& puzzle_file) {
    std::ifstream puzzle_stream(puzzle_file);

    std::vector<PuzzleSpec> puzzles;
    for (std::string line; std::getline(puzzle_stream, line);) {
        const auto result = parse_puzzle_line(line);
        if (result.has_value()) {
            puzzles.push_back(result.value());
        }
    }

    return puzzles;
}
}  // namespace

void solve_puzzles(const std::filesystem::path& puzzle_info_file,
                   const std::filesystem::path& puzzle_file) {
    const auto puzzle_infos = load_puzzle_info(puzzle_info_file);
    const auto puzzles = load_puzzles(puzzle_file);

    for (const auto& spec : puzzles) {
        const auto& puzzle = puzzle_infos.at(spec.type);
        std::cout << "Puzzle " << spec.id << " (" << spec.type << "):" << std::endl;
        std::cout << "initial state:  ";
        for (const auto& idx : spec.initial_state) {
            std::cout << spec.name_from_idx[idx] << " ";
        }
        std::cout << std::endl;
        std::cout << "solution_state: ";
        for (const auto& idx : spec.solution_state) {
            std::cout << spec.name_from_idx[idx] << " ";
        }
        std::cout << std::endl;

        const auto solution =
            solve(puzzle, spec.initial_state, spec.solution_state, spec.num_wild_cards);
        if (solution.has_value()) {
            std::cout << "  Solution: ";
            std::vector<int> state = spec.initial_state;
            for (const auto& action : solution.value()) {
                std::cout << action << " ";
                state = puzzle.actions.at(action)(state);
            }
            std::cout << std::endl;

            std::cout << "solved state:   ";
            for (const auto& idx : state) {
                std::cout << spec.name_from_idx[idx] << " ";
            }
            std::cout << std::endl;

        } else {
            std::cout << "  No solution found" << std::endl;
        }
    }
}
}  // namespace robot::experimental::polytope

int main(int argc, char** argv) {
    cxxopts::Options options("polytope", "A tool solving Kaggle 2023 polytope problem");
    options.add_options()("puzzle_info", "Puzzle info file", cxxopts::value<std::string>())(
        "puzzles", "puzzle file", cxxopts::value<std::string>())("h,help", "Print usage");

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!args.count("puzzle_info")) {
        std::cout << "Missing puzzle info file" << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }

    if (!args.count("puzzles")) {
        std::cout << "Missing puzzle file" << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }

    robot::experimental::polytope::solve_puzzles(args["puzzle_info"].as<std::string>(),
                                                 args["puzzles"].as<std::string>());
    return 0;
}