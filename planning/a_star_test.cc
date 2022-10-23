
#include "planning/a_star.hh"

#include <string>
#include <tuple>

#include "gtest/gtest.h"

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
}  // namespace robot::planning
