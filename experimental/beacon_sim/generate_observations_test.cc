
#include "experimental/beacon_sim/generate_observations.hh"

#include <numbers>

#include "gtest/gtest.h"

namespace experimental::beacon_sim {
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
    constexpr Beacon BEACON = {.id = 123, .pos_x_m = 7.0, .pos_y_m = 8.0};
    constexpr ObservationConfig CONFIG{};

    // ACTION
    const auto maybe_observation = generate_observation(BEACON, robot_state, CONFIG);

    // VERIFICATION
    constexpr double EXPECTED_RANGE_M = 2 * std::numbers::sqrt2;
    constexpr double EXPECTED_BEARING_RAD = -std::numbers::pi / 4.0;
    constexpr double TOL = 1e-6;
    ASSERT_TRUE(maybe_observation.has_value());
    ASSERT_TRUE(maybe_observation->maybe_id.has_value());
    EXPECT_EQ(maybe_observation->maybe_id.value(), BEACON.id);
    ASSERT_TRUE(maybe_observation->maybe_range_m.has_value());
    EXPECT_NEAR(maybe_observation->maybe_range_m.value(), EXPECTED_RANGE_M, TOL);
    ASSERT_TRUE(maybe_observation->maybe_bearing_rad.has_value());
    EXPECT_NEAR(maybe_observation->maybe_bearing_rad.value(), EXPECTED_BEARING_RAD, TOL);
}
}  // namespace experimental::beacon_sim
