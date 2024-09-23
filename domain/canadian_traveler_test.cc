
#include "domain/canadian_traveler.hh"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace robot::domain {
namespace {

using CTG = CanadianTravelerGraph;

std::shared_ptr<CTG> create_test_graph() {
    // Contruct a CTP graph with the following connectivity
    // Note that nodes 0 and 5 are not connected and the
    // edge between 3 and 4 is probabilistically traversable.
    //   ┌───── 2 ─────┐
    //   │             │
    //   │             │
    //   │             │
    //   1      5      4     0
    //   │             :
    //   │             :
    //   │             :
    //   └───── 3 .....:

    const std::vector<CTG::Node> nodes = {
        {{15.0, 0.0}}, {{0.0, 0.0}}, {{5.0, 5.0}}, {{5.0, -5.0}}, {{0.0, 10.0}}, {{0.0, 5.0}},
    };

    const std::vector<CTG::Edge> edges = {
        {1, 2, 2.0, std::nullopt},
        {2, 4, 4.0, std::nullopt},
        {4, 3, 6.0, std::make_optional(0.25)},
        {3, 1, 8.0, std::nullopt},
    };
    return CTG::create(nodes, edges);
}
}  // namespace

TEST(CanadianTravelerTest, graph_constructed_correctly) {
    // Action
    const auto graph = create_test_graph();

    // Verification
    EXPECT_THAT(graph->neighbors(0), testing::IsEmpty());
    EXPECT_THAT(graph->neighbors(1),
                testing::UnorderedElementsAreArray(
                    {CanadianTravelerGraph::Edge{1, 2, 2.0, {}}, {1, 3, 8.0, {}}}));
    EXPECT_THAT(graph->neighbors(2),
                testing::UnorderedElementsAreArray(
                    {CanadianTravelerGraph::Edge{2, 1, 2.0, {}}, {2, 4, 4.0, {}}}));
    EXPECT_THAT(graph->neighbors(3),
                testing::UnorderedElementsAreArray(
                    {CanadianTravelerGraph::Edge{3, 1, 8.0, {}}, {3, 4, 6.0, 0.25}}));
    EXPECT_THAT(graph->neighbors(4),
                testing::UnorderedElementsAreArray(
                    {CanadianTravelerGraph::Edge{4, 2, 4.0, {}}, {4, 3, 6.0, 0.25}}));

    EXPECT_THAT(graph->neighbors(5), testing::IsEmpty());
}

TEST(CanadianTravelerTest, traversable_edge_is_neighbor) {
    // Setup
    const auto graph = create_test_graph();
    const CTG::Weather weather = graph->create_weather(
        {{.id_a = 3, .id_b = 4, .traversability = CTG::EdgeState::Traversable}});

    // Action + Verification
    EXPECT_THAT(weather.neighbors(3), testing::Contains(CTG::Edge{3, 4, 6.0, {}}));
}

TEST(CanadianTravelerTest, untraversable_edge_is_not_neighbor) {
    // Setup
    const auto graph = create_test_graph();
    const CTG::Weather weather = graph->create_weather(
        {{.id_a = 4, .id_b = 3, .traversability = CTG::EdgeState::Untraversable}});
    // Action + Verification
    EXPECT_EQ(weather.neighbors(3).size(), 1);
    EXPECT_THAT(weather.neighbors(3), testing::Contains(CTG::Edge{3, 1, 8.0, {}}));
}

}  // namespace robot::domain
