
#include "experimental/beacon_sim/generate_observations.hh"

#include <numbers>

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(GenerateObservationsTest, single_beacon_test) {
    // SETUP
    //                                    ┌─┐
    //                                    └─┘
    //                                     7,8
    //
    //
    //                   ▲+X
    //                   │
    //                ┌──┼──┐
    //                │  │  │
    //              ◄─┼──┘  │
    //             +Y │  5,6│
    //                └─────┘
    //
    //
    //
    // +Y ▲
    //    │
    //    │
    //    └────►
    //         +X
    // We expect the beacon to be 2*sqrt(2) meters away and at a bearing of -pi/4.0 rad.

    const RobotState robot_state(5.0, 6.0, std::numbers::pi / 2.0);
    const Beacon beacon = {.id = 123, .pos_in_local = {7.0, 8.0}};
    constexpr ObservationConfig CONFIG{};
    std::mt19937 gen;

    // ACTION
    const auto maybe_observation =
        generate_observation(beacon, robot_state, CONFIG, make_in_out(gen));

    // VERIFICATION
    constexpr double EXPECTED_RANGE_M = 2 * std::numbers::sqrt2;
    constexpr double EXPECTED_BEARING_RAD = -std::numbers::pi / 4.0;
    constexpr double TOL = 1e-6;
    ASSERT_TRUE(maybe_observation.has_value());
    ASSERT_TRUE(maybe_observation->maybe_id.has_value());
    EXPECT_EQ(maybe_observation->maybe_id.value(), beacon.id);
    ASSERT_TRUE(maybe_observation->maybe_range_m.has_value());
    EXPECT_NEAR(maybe_observation->maybe_range_m.value(), EXPECTED_RANGE_M, TOL);
    ASSERT_TRUE(maybe_observation->maybe_bearing_rad.has_value());
    EXPECT_NEAR(maybe_observation->maybe_bearing_rad.value(), EXPECTED_BEARING_RAD, TOL);
}

TEST(GenerateObservationsTest, out_of_range_single_beacon_test) {
    const RobotState robot_state(5.0, 6.0, std::numbers::pi / 2.0);
    const Beacon beacon = {.id = 123, .pos_in_local = {7.0, 8.0}};
    constexpr ObservationConfig CONFIG{
        .range_noise_std_m = std::nullopt,
        .max_sensor_range_m = 2.0,
    };
    std::mt19937 gen;

    // ACTION
    const auto maybe_observation =
        generate_observation(beacon, robot_state, CONFIG, make_in_out(gen));

    // VERIFICATION
    EXPECT_FALSE(maybe_observation.has_value());
}
}  // namespace robot::experimental::beacon_sim
