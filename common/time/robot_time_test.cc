
#include "common/time/robot_time.hh"

#include <chrono>
#include <thread>

#include "common/time/sim_clock.hh"
#include "gtest/gtest.h"

using namespace std::literals::chrono_literals;

namespace robot::time {
TEST(RobotTimeTest, default_construction) {
    // Setup + Action
    const RobotTimestamp default_timestamp;
    // Verification
    EXPECT_EQ(default_timestamp.time_since_epoch(), 0s);
}

TEST(RobotTimeTest, inplace_add_subtract) {
    // Setup
    constexpr RobotTimestamp::duration DT_1 = 25ms;
    constexpr RobotTimestamp::duration DT_2 = -75ms;
    {
        // Action
        RobotTimestamp t;
        t += DT_1;

        // Verification
        EXPECT_EQ(t.time_since_epoch(), DT_1);
    }
    {
        // Action
        RobotTimestamp t;
        t -= DT_2;

        // Verification
        EXPECT_EQ(t.time_since_epoch(), -DT_2);
    }
}

TEST(RobotTimeTest, operate_on_left) {
    // Setup
    constexpr RobotTimestamp::duration DT = 25ms;

    {
        // Action
        constexpr RobotTimestamp t1 = RobotTimestamp() + DT;
        constexpr RobotTimestamp t2 = t1 + DT;

        // Verification
        EXPECT_EQ(t1.time_since_epoch(), DT);
        EXPECT_EQ(t2.time_since_epoch(), DT + DT);
    }
    {
        // Action
        constexpr RobotTimestamp t = RobotTimestamp() + DT;

        // Verification
        EXPECT_EQ(t.time_since_epoch(), DT);
    }
    {
        // Action
        constexpr RobotTimestamp t = RobotTimestamp() - DT;

        // Verification
        EXPECT_EQ(t.time_since_epoch(), -DT);
    }
}

TEST(RobotTimeTest, operate_on_right) {
    // Setup
    constexpr RobotTimestamp::duration DT = 25ms;

    {
        // Action
        constexpr RobotTimestamp t = DT + RobotTimestamp();

        // Verification
        EXPECT_EQ(t.time_since_epoch(), DT);
    }
    {
        // Action
        constexpr RobotTimestamp t = DT - RobotTimestamp();

        // Verification
        EXPECT_EQ(t.time_since_epoch(), -DT);
    }
}

TEST(RobotTimeTest, logical_operator_equal) {
    // Setup
    constexpr RobotTimestamp::duration DT_1 = 25ms;
    constexpr RobotTimestamp::duration DT_2 = 100ms;

    // Action + Verification
    EXPECT_EQ(DT_1 * 4 + RobotTimestamp(), DT_2 + RobotTimestamp());
}

TEST(RobotTimeTest, logical_operator_not_equal) {
    // Setup
    constexpr RobotTimestamp::duration DT_1 = 25ms;
    constexpr RobotTimestamp::duration DT_2 = 100ms;

    // Action + Verification
    EXPECT_NE(DT_1 + RobotTimestamp(), DT_2 + RobotTimestamp());
}

TEST(RobotTimeTest, logical_operator_greater_than_less_than) {
    // Setup
    constexpr RobotTimestamp::duration DT_1 = 25ms;
    constexpr RobotTimestamp::duration DT_2 = 100ms;

    // Action + Verification
    EXPECT_LT(DT_1 + RobotTimestamp(), DT_2 + RobotTimestamp());
    EXPECT_GT(DT_2 + RobotTimestamp(), DT_1 + RobotTimestamp());
}

TEST(RobotTimeTest, logical_operator_greater_than_less_than_or_equal) {
    // Setup
    constexpr RobotTimestamp::duration DT_1 = 25ms;
    constexpr RobotTimestamp::duration DT_2 = 100ms;

    // Action + Verification
    EXPECT_LE(DT_1 + RobotTimestamp(), DT_2 + RobotTimestamp());
    EXPECT_GE(DT_2 + RobotTimestamp(), DT_1 + RobotTimestamp());
    EXPECT_LE(DT_1 + RobotTimestamp(), DT_1 + RobotTimestamp());
    EXPECT_GE(DT_2 + RobotTimestamp(), DT_2 + RobotTimestamp());
}

TEST(RobotTimeTest, current_time_steady) {
    // Setup
    set_default_time_provider(TimeProvider::STEADY);
    constexpr auto SLEEP_DT = 100ms;

    // Action
    const RobotTimestamp t1 = current_robot_time();
    std::this_thread::sleep_for(SLEEP_DT);
    const RobotTimestamp t2 = current_robot_time();

    // Verification
    EXPECT_LT(t1, t2);
    EXPECT_LE(t1 + SLEEP_DT, t2);
}

TEST(RobotTimeTest, current_time_sim) {
    // Setup
    set_default_time_provider(TimeProvider::SIM);
    constexpr auto SLEEP_DT = 100ms;
    constexpr auto ADVANCE_DT = 150ms;

    // Action
    const RobotTimestamp t1 = current_robot_time();
    std::this_thread::sleep_for(SLEEP_DT);
    const RobotTimestamp t2 = current_robot_time();
    SimClock::advance(ADVANCE_DT);
    const RobotTimestamp t3 = current_robot_time();

    // Verification
    EXPECT_EQ(t1, RobotTimestamp());
    EXPECT_EQ(t1, t2);
    EXPECT_LT(t1, t3);
    EXPECT_EQ(t3 - t1, ADVANCE_DT);
}

}  // namespace robot::time
