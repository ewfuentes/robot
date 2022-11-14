
#include "planning/belief_road_map.hh"

#include "gtest/gtest.h"

namespace robot::planning {
namespace {
constexpr double NOISE_PER_M = 0.1;
constexpr double NOISE_REDUCTION = 0.1;
BeliefUpdater make_belief_updater(const RoadMap &road_map, const Eigen::Vector2d &goal_state) {
    return [&road_map, &goal_state](const Belief &initial_belief, const int start_idx,
                                    const int end_idx) {
        const bool should_decrease =
            (start_idx == 0 || start_idx == 1) && (end_idx == 0 || end_idx == 1);
        const double cov_mult = should_decrease ? NOISE_REDUCTION : 1.0;
        const Eigen::Vector2d new_mean = end_idx >= 0 ? road_map.points.at(end_idx) : goal_state;

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

    // Since measurements occur along both directions of the (0, 1) edge,
    // it is expected that the covariance will converge after repeated traversals.
    // at convergence, it is expected that the following relation will hold:
    // cov_{t+1} = (cov_t + NOISE_PER_M * dist_m) * NOISE_REDUCTION
    // cov_t = NOISE_REDUCTION * NOISE_PER_M * dist_m / (1 - NOISE_REDUCTION)
    const double dist_m = (road_map.points.at(0) - road_map.points.at(1)).norm();
    const double min_cov = NOISE_REDUCTION * NOISE_PER_M * dist_m / (1 - NOISE_REDUCTION);

    // After min_cov is reached, no futher measurements are accumulated along the (1,2) and (2,
    // goal) edges. Add the accumulated noise to the min covariance to get the expected covariance.
    const double accumulated_dist_m = (road_map.points.at(1) - road_map.points.at(2)).norm() +
                                      (road_map.points.at(2) - goal_state).norm();
    const double accumulated_noise = NOISE_PER_M * accumulated_dist_m;
    const double expected_cov = min_cov + accumulated_noise;

    // Action
    const auto maybe_brm_plan =
        plan(road_map, initial_belief, make_belief_updater(road_map, goal_state), goal_state);

    // Verification
    EXPECT_TRUE(maybe_brm_plan.has_value());
    const auto &plan = maybe_brm_plan.value();

    constexpr double TOL = 1e-6;
    EXPECT_EQ(plan.nodes.front(), BRMPlan::INITIAL_BELIEF_NODE_IDX);
    EXPECT_EQ(plan.nodes.back(), BRMPlan::GOAL_BELIEF_NODE_IDX);
    EXPECT_EQ(plan.beliefs.back().mean, goal_state);
    EXPECT_NEAR(plan.beliefs.back().cov(0, 0), expected_cov, TOL);
    EXPECT_NEAR(plan.beliefs.back().cov(1, 1), expected_cov, TOL);
    EXPECT_NEAR(plan.beliefs.back().cov(0, 1), 0.0, TOL);
    EXPECT_NEAR(plan.beliefs.back().cov(1, 0), 0.0, TOL);
}
}  // namespace robot::planning
