
#include "experimental/ctp/compute_optimistic_rollout.hh"

#include "domain/canadian_traveler.hh"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace robot::experimental::ctp {
namespace {
using CTG = domain::CanadianTravelerGraph;

constexpr int START_IDX = 0;
constexpr int A_IDX = 1;
constexpr int B_IDX = 2;
constexpr int GOAL_IDX = 3;
std::shared_ptr<CTG> create_test_graph() {
    // Create the following graph. There exists a known traversable path from the start
    // to the goal. There also exists a lower cost probabilistically traversable path
    // that may be cheaper.
    //
    //      S   C=1
    //      ├─────────┐A
    //      │         :
    //  C=10│         : C=1
    //      │         : p=0.1
    //      │         :
    //      │         :
    //      │         :
    //      ├─────────┘B
    //      G   C=1
    // Start, A, B, Goal
    std::vector<CTG::Node> nodes = {{{0, 10}}, {{10, 10}}, {{10, 0}}, {{0, 0}}};

    std::vector<CTG::Edge> edges = {
        {.node_a = START_IDX, .node_b = GOAL_IDX, .cost = 10.0, .traversal_prob = {}},
        {.node_a = START_IDX, .node_b = A_IDX, .cost = 1.0, .traversal_prob = {}},
        {.node_a = A_IDX, .node_b = B_IDX, .cost = 1.0, .traversal_prob = 0.1},
        {.node_a = B_IDX, .node_b = GOAL_IDX, .cost = 1.0, .traversal_prob = {}},
    };

    return CTG::create(std::move(nodes), std::move(edges));
}
}  // namespace

TEST(ComputeOptimisticRolloutTest, rollout_in_traversable_weather) {
    // Setup
    const auto graph_ptr = create_test_graph();
    const auto weather = graph_ptr->create_weather(
        {{.id_a = A_IDX, .id_b = B_IDX, .traversability = CTG::EdgeState::Traversable}});

    // Action
    const auto rollout = compute_optimistic_rollout(*graph_ptr, START_IDX, GOAL_IDX, weather);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_THAT(rollout.path, testing::ElementsAre(START_IDX, A_IDX, B_IDX, GOAL_IDX));
    EXPECT_NEAR(rollout.cost, 3.0, TOL);
}

TEST(ComputeOptimisticRolloutTest, rollout_in_untraversable_weather) {
    // Setup
    const auto graph_ptr = create_test_graph();
    const auto weather = graph_ptr->create_weather(
        {{.id_a = A_IDX, .id_b = B_IDX, .traversability = CTG::EdgeState::Untraversable}});

    // Action
    const auto rollout = compute_optimistic_rollout(*graph_ptr, START_IDX, GOAL_IDX, weather);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_THAT(rollout.path, testing::ElementsAre(START_IDX, A_IDX, START_IDX, GOAL_IDX));
    EXPECT_NEAR(rollout.cost, 12.0, TOL);
}
}  // namespace robot::experimental::ctp
