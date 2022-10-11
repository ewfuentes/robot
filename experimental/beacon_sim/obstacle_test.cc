
#include "experimental/beacon_sim/obstacle.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(ObstacleTest, SquareTest) {
    // Setup
    const Eigen::Vector2d LOWER_LEFT{{12.0, 23.0}};
    const double SIDE_LENGTH = 20;
    const std::vector<Eigen::Vector2d> CORNERS{
        SIDE_LENGTH * Eigen::Vector2d::Zero() + LOWER_LEFT,
        SIDE_LENGTH * Eigen::Vector2d::UnitY() + LOWER_LEFT,
        SIDE_LENGTH * Eigen::Vector2d::Ones() + LOWER_LEFT,
        SIDE_LENGTH * Eigen::Vector2d::UnitX() + LOWER_LEFT,
    };
    const Obstacle obstacle(CORNERS);

    // List of query points and expected inside value
    const std::array<std::pair<Eigen::Vector2d, bool>, 4> QUERIES{{
        {Eigen::Vector2d{0, 0}, false},
        {Eigen::Vector2d{13.0, 24.0}, true},
        {Eigen::Vector2d{11.0, 24.0}, false},
        {Eigen::Vector2d{100.0, 100.0}, false},
    }};

    // Action + Verification
    for (const auto &[query_pt, expected_answer] : QUERIES) {
        EXPECT_EQ(obstacle.is_inside(query_pt), expected_answer);
    }
}
}  // namespace robot::experimental::beacon_sim
