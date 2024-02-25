
#include "planning/djikstra.hh"

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
TEST(DjikstraTest, graph_test) {
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

    auto termination_check = [](std::unordered_map<std::string, double>) { return false; };

    // ACTION
    const auto result = djikstra<std::string>("A", successor_func, termination_check);

    // VERIFICATION
    EXPECT_EQ(result.cost_to_go_from_state.at("A"), 0.0);
    EXPECT_EQ(result.cost_to_go_from_state.at("B"), 10.0);
    EXPECT_EQ(result.cost_to_go_from_state.at("C"), 5.0);
    EXPECT_EQ(result.cost_to_go_from_state.at("D"), 17.0);

    EXPECT_FALSE(result.back_pointer.contains("A"));
    EXPECT_EQ(result.back_pointer.at("B"), "A");
    EXPECT_EQ(result.back_pointer.at("C"), "A");
    EXPECT_EQ(result.back_pointer.at("D"), "B");

}
}  // namespace robot::planning
