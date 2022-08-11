
#include "experimental/beacon_sim/robot.hh"

#include <numbers>

#include "gtest/gtest.h"

namespace experimental::beacon_sim {
TEST(RobotTest, robot_moves_as_expected) {
    // SETUP
    constexpr double TOL = 1e-6;
    constexpr double MOVE_DIST_M = 1.5;
    const RobotState init_state(2.0, 3.0, 0.0);
    // ACTION
    auto new_state = init_state;
    new_state.move(MOVE_DIST_M);

    // VERIFICATION
    EXPECT_NEAR(new_state.pos_x_m(), init_state.pos_x_m() + MOVE_DIST_M, TOL);
    EXPECT_NEAR(new_state.pos_y_m(), init_state.pos_y_m(), TOL);
    EXPECT_NEAR(new_state.heading_rad(), init_state.heading_rad(), TOL);
    EXPECT_TRUE(false);
}

TEST(RobotTest, robot_turns_as_expected) {
    // SETUP
    constexpr double TOL = 1e-6;
    constexpr double TURN_RAD = 1.5;
    const RobotState init_state(2.0, 3.0, 0.0);
    // ACTION
    auto new_state = init_state;
    new_state.turn(TURN_RAD);

    // VERIFICATION
    EXPECT_NEAR(new_state.pos_x_m(), init_state.pos_x_m(), TOL);
    EXPECT_NEAR(new_state.pos_y_m(), init_state.pos_y_m(), TOL);
    EXPECT_NEAR(new_state.heading_rad(), init_state.heading_rad() + TURN_RAD, TOL);
}

TEST(RobotTest, robot_moves_in_square) {
    // SETUP
    constexpr double TOL = 1e-6;
    constexpr double MOVE_DIST_M = 1.23;
    constexpr double TURN_RAD = std::numbers::pi / 2.0;
    const RobotState init_state(2.0, 3.0, 1.0);
    // ACTION
    auto new_state = init_state;
    for (int i = 0; i < 4; i++) {
        new_state.move(MOVE_DIST_M);
        new_state.turn(TURN_RAD);
    }

    // VERIFICATION
    EXPECT_NEAR(new_state.pos_x_m(), init_state.pos_x_m(), TOL);
    EXPECT_NEAR(new_state.pos_y_m(), init_state.pos_y_m(), TOL);
    EXPECT_NEAR(new_state.heading_rad(), init_state.heading_rad() + 2 * std::numbers::pi, TOL);
}
}  // namespace experimental::beacon_sim
