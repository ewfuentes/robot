
#include "planning/belief_road_map.hh"

#include "gtest/gtest.h"

namespace robot::planning {
namespace {
BeliefUpdater make_belief_updater(const RoadMap &road_map, const Eigen::Vector2d &goal_state) {
    return [&road_map, &goal_state](const Belief &initial_belief, const int start_idx,
                                    const int end_idx) {
        constexpr double NOISE_PER_M = 0.1;
        const bool should_decrease =
            (start_idx == 0 || start_idx == 1) && (end_idx == 0 || end_idx == 1);
        const double cov_mult = should_decrease ? 0.1 : 1.0;
        const Eigen::Vector2d new_mean = end_idx > 0 ? road_map.points.at(end_idx) : goal_state;

        const double dist_m = (initial_belief.mean - new_mean).norm();

        return Belief{
            .mean = new_mean,
            .cov = (initial_belief.cov + Eigen::Matrix2d::Identity() * NOISE_PER_M * dist_m) *
                   cov_mult,
        };
    };
}
}  // namespace

TEST(BeliefRoadMapTest, linear_graph) {
    // Setup
    // Create a graph A - B - C where to robot starts at B and the goal is at C.
    // The robot receives measurements along the edge (A, B), reducing it's uncertainty.
    // The process noise is such that moving from B to A, then to B and C results in a tighter
    // covariance at the goal.

    const RoadMap road_map = {
        .points = {{-10.0, 0.0}, {0.0, 0.0}, {10.0, 0.0}},
        .adj = Eigen::Matrix3d{{{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}}},
    };

    const Belief initial_belief = {
        .mean = {0.0, 3.0},
        .cov = Eigen::Matrix2d{{3.0, 0.0}, {0.0, 3.0}},
    };

    const Eigen::Vector2d goal_state{12.0, 0.0};

    // Action
    const auto maybe_brm_plan =
        plan(road_map, initial_belief, make_belief_updater(road_map, goal_state), goal_state);

    // Verification
    EXPECT_TRUE(maybe_brm_plan.has_value());
    EXPECT_EQ(maybe_brm_plan->nodes.size(), 5);
    std::cout << "Plan: " << std::endl << maybe_brm_plan.value() << std::endl;
}
}  // namespace robot::planning
