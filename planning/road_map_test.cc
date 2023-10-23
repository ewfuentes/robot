
#include "planning/road_map.hh"

#include <optional>
#include <stdexcept>

#include "common/check.hh"
#include "gtest/gtest.h"

namespace robot::planning {
TEST(RoadMapTest, no_start_neighbors_throws) {
    // Setup
    const std::vector<Eigen::Vector2d> pts = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    const Eigen::MatrixXd adj = (Eigen::MatrixXd(3, 3) << 0, 1, 1, 1, 0, 1, 1, 1, 0).finished();

    const Eigen::Vector2d start{10.0, 20.0};
    const Eigen::Vector2d goal{2.0, 3.0};
    constexpr double CONNECTION_RADIUS_M = 2;

    // Action + Verification
    EXPECT_THROW(
        RoadMap(pts, adj,
                {{.start = start, .goal = goal, .connection_radius_m = CONNECTION_RADIUS_M}}),
        robot::check_failure);
}

TEST(RoadMapTest, no_goal_neighbors_throws) {
    // Setup
    const std::vector<Eigen::Vector2d> pts = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    const Eigen::MatrixXd adj = (Eigen::MatrixXd(3, 3) << 0, 1, 1, 1, 0, 1, 1, 1, 0).finished();

    const Eigen::Vector2d start{0.0, 0.0};
    const Eigen::Vector2d goal{20.0, 30.0};
    constexpr double CONNECTION_RADIUS_M = 3.0;

    // Action + Verification
    EXPECT_THROW(
        RoadMap(pts, adj,
                {{.start = start, .goal = goal, .connection_radius_m = CONNECTION_RADIUS_M}}),
        robot::check_failure);
}

TEST(RoadMapTest, invalid_index_throws) {
    // Setup
    const std::vector<Eigen::Vector2d> pts = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    const Eigen::MatrixXd adj = (Eigen::MatrixXd(3, 3) << 0, 1, 1, 1, 0, 1, 1, 1, 0).finished();
    const RoadMap road_map(pts, adj);

    // Action + Verification
    EXPECT_THROW(road_map.point(10), std::out_of_range);
}

TEST(RoadMapTest, goal_index_when_no_goal_set_throws) {
    // Setup
    const std::vector<Eigen::Vector2d> pts = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    const Eigen::MatrixXd adj = (Eigen::MatrixXd(3, 3) << 0, 1, 1, 1, 0, 1, 1, 1, 0).finished();
    const RoadMap road_map(pts, adj);

    // Action + Verification
    EXPECT_THROW(road_map.point(RoadMap::GOAL_IDX), std::bad_optional_access);
}

TEST(RoadMapTest, start_index_when_no_start_set_throws) {
    // Setup
    const std::vector<Eigen::Vector2d> pts = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    const Eigen::MatrixXd adj = (Eigen::MatrixXd(3, 3) << 0, 1, 1, 1, 0, 1, 1, 1, 0).finished();
    const RoadMap road_map(pts, adj);

    // Action + Verification
    EXPECT_THROW(road_map.point(RoadMap::START_IDX), std::bad_optional_access);
}
}  // namespace robot::planning
