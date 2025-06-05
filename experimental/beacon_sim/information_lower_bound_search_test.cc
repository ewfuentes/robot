
#include "experimental/beacon_sim/information_lower_bound_search.hh"

#include "gtest/gtest.h"
#include "planning/road_map.hh"

namespace robot::experimental::beacon_sim {
namespace {

constexpr int START_IDX = planning::RoadMap::START_IDX;
constexpr int GOAL_IDX = planning::RoadMap::GOAL_IDX;
constexpr int FORK_IDX = 0;
constexpr int JOIN_IDX = 1;
constexpr int MEASUREMENT_IDX = 2;
constexpr int FAR_NODE_IDX = 3;

struct TestEnvironment {
    planning::RoadMap road_map;
    LowerBoundReversePropagator rev_prop;
};

planning::RoadMap create_triangle_road_map() {
    return planning::RoadMap({{1, 0}, {2, 0}, {1.5, 1}, {1.5, -10}},
                             // clang-format off
        (Eigen::MatrixXd(4, 4) << 0, 1, 1, 1, 
                                  1, 0, 1, 1, 
                                  1, 1, 0, 0, 
                                  1, 1, 0, 0).finished(),
                             // clang-format on
                             {{
                                 .start = {0.0, 0.0},
                                 .goal = {3.0, 0.0},
                                 .connection_radius_m = 1.1,
                             }});
}

LowerBoundReversePropagator create_triangle_rev_prop() {
    return [](const int start, const int end, const double bound_at_end) -> PropagationResult {
        const double Q = 0.2;
        const double M = start == MEASUREMENT_IDX || end == MEASUREMENT_IDX ? 0.4 : 0;
        const double info_lower_bound = (bound_at_end - M) / (1 - Q * (bound_at_end - M));
        return {
            .info_lower_bound = info_lower_bound,
            .edge_cost = (start == FAR_NODE_IDX || end == FAR_NODE_IDX) ? 10.0 : 1.0,
        };
    };
}

TestEnvironment create_triangle_test_env() {
    // Create the triangle test environment below
    // We start on the left and want to get to the right. The path through 1-2
    // does not have any beacons where as the 1-3-2 path does. The optimal behavior
    // is if the belief has more information than some amount, to proceed directly
    // across the 1-2 path. However, if there is less information than this threshold
    // amount, the 1-3-2 path should be desirable. The 1-4-2 path has no beacons and
    // is significantly longer than 1-2 or 1-3-2, so it should never be preferred.
    //
    //               B
    //
    //               2
    //           ┌───X───┐
    //           │       │
    //       c=1 │       │ c=1
    //           │0     1│
    // Start ────X───────X──── Goal
    //           │  c=1  │
    //           │       │
    //      c=10 │       │ c=10
    //           └───X───┘
    //               3
    //

    return {
        .road_map = create_triangle_road_map(),
        .rev_prop = create_triangle_rev_prop(),
    };
}
}  // namespace

TEST(InformationLowerBoundSearch, grid_environment_search_with_low_info_start) {
    // Setup
    TestEnvironment env = create_triangle_test_env();
    constexpr double GOAL_INFO_LOWER_BOUND = 1;
    constexpr double START_INFO = 1;

    // Action
    const auto result = information_lower_bound_search(env.road_map, START_INFO,
                                                       GOAL_INFO_LOWER_BOUND, env.rev_prop);

    // Verification
    // When starting with low information, it is best to divert toward the landmark.
    EXPECT_EQ(result.path_to_goal.size(), 5);
    EXPECT_EQ(result.path_to_goal.at(0), START_IDX);
    EXPECT_EQ(result.path_to_goal.at(1), FORK_IDX);
    EXPECT_EQ(result.path_to_goal.at(2), MEASUREMENT_IDX);
    EXPECT_EQ(result.path_to_goal.at(3), JOIN_IDX);
    EXPECT_EQ(result.path_to_goal.at(4), GOAL_IDX);
}

TEST(InformationLowerBoundSearch, grid_environment_search_with_high_info_start) {
    // Setup
    TestEnvironment env = create_triangle_test_env();
    constexpr double GOAL_INFO_LOWER_BOUND = 1;
    constexpr double START_INFO = 10;

    // Action
    const auto result = information_lower_bound_search(env.road_map, START_INFO,
                                                       GOAL_INFO_LOWER_BOUND, env.rev_prop);

    // Verification
    // When starting with high information, it is best to go straight to the goal;
    EXPECT_EQ(result.path_to_goal.size(), 4);
    EXPECT_EQ(result.path_to_goal.at(0), START_IDX);
    EXPECT_EQ(result.path_to_goal.at(1), FORK_IDX);
    EXPECT_EQ(result.path_to_goal.at(2), JOIN_IDX);
    EXPECT_EQ(result.path_to_goal.at(3), GOAL_IDX);
}

TEST(InformationLowerBoundSearch, should_merge_with_empty) {
    // Setup
    const std::vector<detail::InProgressPath> existing;
    detail::InProgressPath new_path = {
        .info_lower_bound = 20, .cost_to_go = 10, .path_to_goal = {}};

    // Action
    const detail::MergeResult result = should_merge(existing, new_path);

    // Verification
    EXPECT_TRUE(result.should_merge);
    EXPECT_TRUE(result.dominated_paths_idxs.empty());
}

TEST(InformationLowerBoundSearch, dominated_paths_are_booted) {
    // Setup
    const std::vector<detail::InProgressPath> existing = {
        {
            .info_lower_bound = 30,
            .cost_to_go = 40,
            .path_to_goal = {},
        },
        {
            .info_lower_bound = 40,
            .cost_to_go = 15,
            .path_to_goal = {},
        },
        {
            .info_lower_bound = 10,
            .cost_to_go = 50,
            .path_to_goal = {},
        },
    };
    detail::InProgressPath new_path = {
        .info_lower_bound = 20, .cost_to_go = 10, .path_to_goal = {}};

    // Action
    const detail::MergeResult result = should_merge(existing, new_path);

    // Verification
    EXPECT_TRUE(result.should_merge);
    EXPECT_EQ(result.dominated_paths_idxs.size(), 2);
    EXPECT_EQ(result.dominated_paths_idxs.at(0), 0);
    EXPECT_EQ(result.dominated_paths_idxs.at(1), 1);
}

TEST(InformationLowerBoundSearch, dominated_new_path_is_dropped) {
    // Setup
    const std::vector<detail::InProgressPath> existing = {
        {
            .info_lower_bound = 30,
            .cost_to_go = 40,
            .path_to_goal = {},
        },
        {
            .info_lower_bound = 40,
            .cost_to_go = 15,
            .path_to_goal = {},
        },
        {
            .info_lower_bound = 10,
            .cost_to_go = 50,
            .path_to_goal = {},
        },
    };
    detail::InProgressPath new_path = {
        .info_lower_bound = 15, .cost_to_go = 60, .path_to_goal = {}};

    // Action
    const detail::MergeResult result = should_merge(existing, new_path);

    // Verification
    EXPECT_FALSE(result.should_merge);
    EXPECT_TRUE(result.dominated_paths_idxs.empty());
}

TEST(InformationLowerBoundSearch, keep_all_paths_if_different) {
    // Setup
    const std::vector<detail::InProgressPath> existing = {
        {
            .info_lower_bound = 30,
            .cost_to_go = 40,
            .path_to_goal = {},
        },
        {
            .info_lower_bound = 40,
            .cost_to_go = 15,
            .path_to_goal = {},
        },
        {
            .info_lower_bound = 10,
            .cost_to_go = 50,
            .path_to_goal = {},
        },
    };
    detail::InProgressPath new_path = {.info_lower_bound = 5, .cost_to_go = 60, .path_to_goal = {}};

    // Action
    const detail::MergeResult result = should_merge(existing, new_path);

    // Verification
    EXPECT_TRUE(result.should_merge);
    EXPECT_TRUE(result.dominated_paths_idxs.empty());
}
}  // namespace robot::experimental::beacon_sim
