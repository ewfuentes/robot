
#include "planning/a_star.hh"

#include <cmath>
#include <string>
#include <tuple>

#include "gtest/gtest.h"

namespace std {
// Add a std::hash specialization for tuple<int, int>
template <>
struct hash<tuple<int, int>> {
    size_t operator()(const tuple<int, int> &item) const {
        hash<int> int_hasher;
        const auto &[a, b] = item;
        return int_hasher(a) ^ (int_hasher(b) << 4);
    }
};

}  // namespace std

namespace robot::planning {
TEST(AStarTest, uninformed_search) {
    // SETUP
    // Create an undirected diamond graph, where one half is cheaper than the other
    const std::vector<std::tuple<std::string, std::string, double>> graph{
        {"A", "B", 10.0}, {"A", "C", 5.0}, {"B", "D", 7.0}, {"C", "D", 15.0}};

    auto successor_func = [&graph](const std::string &node) -> std::vector<Successor<std::string>> {
        std::vector<Successor<std::string>> out;
        for (const auto &edge : graph) {
            const auto &node_a = std::get<0>(edge);
            const auto &node_b = std::get<1>(edge);
            if (node == std::get<0>(edge) || node == std::get<1>(edge)) {
                const auto &other_node = node == node_a ? node_b : node_a;
                out.push_back({
                    .state = other_node,
                    .edge_cost = std::get<double>(edge),
                });
            }
        }

        return out;
    };

    auto heuristic_func = [](const std::string &) -> double { return 0; };

    auto termination_check = [](const std::string &node) { return node == "D"; };

    // ACTION
    const auto result = a_star<std::string>("A", successor_func, heuristic_func, termination_check);

    // VERIFICATION
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result->states.size(), 3);
    EXPECT_EQ(result->states.at(0), "A");
    EXPECT_EQ(result->states.at(1), "B");
    EXPECT_EQ(result->states.at(2), "D");
    EXPECT_EQ(result->cost, 17);
}

TEST(AStarTest, grid_world) {
    // SETUP
    // Create a grid world with unit move cost
    using Cell = std::tuple<int, int>;
    // Add a goal at x = 10, y = 0
    constexpr Cell GOAL{10, 0};
    constexpr Cell INITIAL_STATE{0, 0};
    auto successor_func = [](const Cell &cell) -> std::vector<Successor<Cell>> {
        const auto [cell_x, cell_y] = cell;
        std::vector<Successor<Cell>> out;
        for (const auto &[delta_x, delta_y] :
             std::array<Cell, 4>{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}}) {
            // Add an infinite wall at x = 5 with a gap at y = 10
            if (cell_x + delta_x == 5 && cell_y + delta_y != 10) {
                continue;
            }
            out.push_back(Successor<Cell>{
                .state = {cell_x + delta_x, cell_y + delta_y},
                .edge_cost = 1.0,
            });
        }

        return out;
    };

    // Manhattan Distance
    auto heuristic_func = [GOAL = GOAL](const Cell &cell) -> double {
        const int delta_x = std::get<0>(GOAL) - std::get<0>(cell);
        const int delta_y = std::get<1>(GOAL) - std::get<1>(cell);
        return std::abs(delta_x) + std::abs(delta_y);
    };

    auto termination_check = [GOAL = GOAL](const Cell &cell) { return cell == GOAL; };

    // ACTION
    const auto result = a_star<Cell>(Cell{0, 0}, successor_func, heuristic_func, termination_check);

    // VERIFICATION
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result->states.size(), 31);
    EXPECT_EQ(result->states.at(0), INITIAL_STATE);
    EXPECT_EQ(result->states.at(30), GOAL);
    EXPECT_EQ(result->states.at(15), (Cell{5, 10}));
}
}  // namespace robot::planning
