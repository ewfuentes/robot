
#include "planning/id_a_star.hh"
#include "gtest/gtest.h"

namespace robot::planning {
namespace {
using NodeId = int;
std::vector<Successor<NodeId>> successors_for_node(const NodeId &node_id) {
    // Create the following graph.
    //       1
    //       ▲
    //       ▼
    //   3◄─►2◄─►4
    const std::unordered_map<NodeId, std::vector<NodeId>> graph = {
        {{1}, {2}}, {{2}, {3, 4}}, {{3}, {2}}, {{4}, {2}}};
    const auto neighbors = graph.at(node_id);
    std::vector<Successor<NodeId>> out;
    for (const auto &neighbor : neighbors) {
        out.push_back({.state = neighbor, .edge_cost = 1.0});
    }
    return out;
}
}

TEST(IDAStarTest, simple_test) {
    // Setup
    constexpr int START_NODE = 1;
    constexpr int GOAL_NODE = 4;

    const auto goal_check = [](const NodeId &node) {
        return node == GOAL_NODE;
    };

    const auto heuristic = [](const NodeId &) {
        return 0.0;
    };

    // Action
    const auto result = id_a_star(START_NODE, successors_for_node,
                                  heuristic, goal_check);

    // Verification

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result->states.size(), 3);
    EXPECT_EQ(result->states[0], 1);
    EXPECT_EQ(result->states[1], 2);
    EXPECT_EQ(result->states[2], 4);
}
}