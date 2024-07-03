
#include "planning/breadth_first_search.hh"

#include <unordered_map>
#include <unordered_set>

#include "gtest/gtest.h"

namespace robot::planning {
namespace {
using NodeId = int;

std::vector<BFSSuccessor<NodeId>> successors_for_node(const NodeId &node_id) {
    // Create the following graph.
    //       1
    //       ▲
    //       ▼
    //   3◄─►2◄─►4
    const std::unordered_map<NodeId, std::vector<NodeId>> graph = {
        {{1}, {2}}, {{2}, {3, 4}}, {{3}, {2}}, {{4}, {2}}};
    const auto neighbors = graph.at(node_id);
    std::vector<BFSSuccessor<NodeId>> out;
    for (const auto &neighbor : neighbors) {
        out.push_back({.state = neighbor, .edge_cost = 1.0});
    }
    return out;
}

}  // namespace

TEST(BreadthFirstSearchTest, acceptable_path_returned_immediately) {
    // Setup
    constexpr int START_NODE = 1;
    constexpr int GOAL_NODE = 4;
    std::unordered_set<NodeId> visited_list;
    ShouldQueueFunc<NodeId> have_not_visited_before =
        [&visited_list](const BFSSuccessor<NodeId> &to_add, const int &,
                        const std::vector<Node<NodeId>> &) -> ShouldQueueResult {
        if (visited_list.contains(to_add.state)) {
            return ShouldQueueResult::SKIP;
        }
        visited_list.insert(to_add.state);
        return ShouldQueueResult::QUEUE;
    };

    const GoalCheckFunc<NodeId> goal_check = [](const Node<NodeId> &node) {
        return node.state == GOAL_NODE;
    };
    const IdentifyPathEndFunc<NodeId> identify_end =
        [&goal_check](const std::vector<Node<NodeId>> &nodes) -> std::optional<int> {
        const auto iter = std::find_if(nodes.begin(), nodes.end(), goal_check);
        if (iter == nodes.end()) {
            return std::nullopt;
        }
        return std::distance(nodes.begin(), iter);
    };

    // Action
    const auto result = breadth_first_search(START_NODE, std::function(successors_for_node),
                                             have_not_visited_before, goal_check, identify_end);

    // Verification

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result->path.size(), 3);
    EXPECT_EQ(result->path[0], 1);
    EXPECT_EQ(result->path[1], 2);
    EXPECT_EQ(result->path[2], 4);
    EXPECT_EQ(result->num_nodes_expanded, 4);
    EXPECT_EQ(result->num_nodes_visited, 4);
}

TEST(BreadthFirstSearchTest, find_longest_path_visiting_at_most_twice) {
    // Setup
    constexpr int START_NODE = 1;
    constexpr int GOAL_NODE = 4;
    ShouldQueueFunc<NodeId> have_visited_at_most_once_before =
        [](const BFSSuccessor<NodeId> &to_add, const int &parent_idx,
           const std::vector<Node<NodeId>> &nodes) {
            std::optional<int> node_idx = parent_idx;
            int prev_visit_counts = 0;
            while (node_idx.has_value()) {
                if (nodes[node_idx.value()].state == to_add.state) {
                    prev_visit_counts++;
                }
                if (prev_visit_counts > 1) {
                    return ShouldQueueResult::SKIP;
                }
                node_idx = nodes[node_idx.value()].maybe_parent_idx;
            }

            return ShouldQueueResult::QUEUE;
        };

    const GoalCheckFunc<NodeId> goal_check = [](const Node<NodeId> &) {
        // Don't terminate early
        return false;
    };

    const IdentifyPathEndFunc<NodeId> identify_end =
        [](const std::vector<Node<NodeId>> &nodes) -> std::optional<int> {
        const auto iter = std::max_element(nodes.begin(), nodes.end(),
                                           [](const Node<NodeId> &a, const Node<NodeId> &b) {
                                               const bool is_a_goal = a.state == GOAL_NODE;
                                               const bool is_b_goal = b.state == GOAL_NODE;
                                               if (is_a_goal && is_b_goal) {
                                                   // If both nodes are at the goal, compare based
                                                   // on cost
                                                   return a.cost < b.cost;
                                               } else if (is_a_goal) {
                                                   // If only a is the goal, place it higher than
                                                   // node b
                                                   return false;
                                               } else {
                                                   // Either b is the goal or neither a nor b is at
                                                   // the goal
                                                   return true;
                                               }
                                           });
        if (iter == nodes.end()) {
            return std::nullopt;
        }
        return std::distance(nodes.begin(), iter);
    };

    // Action
    const auto result =
        breadth_first_search(START_NODE, std::function(successors_for_node),
                             have_visited_at_most_once_before, goal_check, identify_end);

    // Verification

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result->path.size(), 5);
    EXPECT_EQ(result->path[0], 1);
    EXPECT_EQ(result->path[1], 2);
    EXPECT_EQ(result->path[2], 3);
    EXPECT_EQ(result->path[3], 2);
    EXPECT_EQ(result->path[4], 4);
}
}  // namespace robot::planning
