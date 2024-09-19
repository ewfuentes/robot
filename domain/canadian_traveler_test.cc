
#include "domain/canadian_traveler.hh"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace robot::domain {
TEST(CanadianTravelerTest, GraphConstructedCorrectly) {
    // Setup
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

    const std::vector<CanadianTravelerGraph::Node> nodes = {
        {{15.0, 0.0}}, {{0.0, 0.0}}, {{5.0, 5.0}}, {{5.0, -5.0}}, {{0.0, 10.0}}, {{0.0, 5.0}},
    };

    const std::vector<CanadianTravelerGraph::Edge> edges = {
        {1, 2, 2.0, std::nullopt},
        {2, 4, 4.0, std::nullopt},
        {4, 3, 6.0, std::make_optional(0.25)},
        {3, 1, 8.0, std::nullopt},
    };

    // Action
    const CanadianTravelerGraph graph(nodes, edges);

    // Verification
    EXPECT_THAT(graph.neighbors(0), testing::IsEmpty());
    EXPECT_THAT(graph.neighbors(1),
                testing::UnorderedElementsAreArray(
                    {CanadianTravelerGraph::Edge{1, 2, 2.0, {}}, {1, 3, 8.0, {}}}));
    EXPECT_THAT(graph.neighbors(2),
                testing::UnorderedElementsAreArray(
                    {CanadianTravelerGraph::Edge{2, 1, 2.0, {}}, {2, 4, 4.0, {}}}));
    EXPECT_THAT(graph.neighbors(3),
                testing::UnorderedElementsAreArray(
                    {CanadianTravelerGraph::Edge{3, 1, 8.0, {}}, {3, 4, 6.0, 0.25}}));
    EXPECT_THAT(graph.neighbors(4),
                testing::UnorderedElementsAreArray(
                    {CanadianTravelerGraph::Edge{4, 2, 4.0, {}}, {4, 3, 6.0, 0.25}}));

    EXPECT_THAT(graph.neighbors(5), testing::IsEmpty());
}
}  // namespace robot::domain
