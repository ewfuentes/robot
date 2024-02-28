
#include "experimental/beacon_sim/world_map.hh"

#include <chrono>

#include "common/time/robot_time.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(BeaconTest, static_beacon_test) {
    // Setup
    const WorldMapConfig config = {
        .fixed_beacons = {.beacons = {{.id = 1, .pos_in_local = {2.0, 3.0}},
                                      {.id = 4, .pos_in_local = {5.0, 6.0}},
                                      {.id = 7, .pos_in_local = {8.0, 9.0}}}},
        .blinking_beacons = {},
        .correlated_beacons = {},
        .obstacles = {},
    };

    // Action
    const WorldMap map(config);
    const auto visible_beacons = map.visible_beacons(time::current_robot_time());

    // Verification
    EXPECT_EQ(visible_beacons.size(), config.fixed_beacons.beacons.size());
    for (const auto &beacon : config.fixed_beacons.beacons) {
        const auto iter =
            std::find_if(visible_beacons.begin(), visible_beacons.end(),
                         [&beacon](const Beacon &other) { return beacon.id == other.id; });
        ASSERT_NE(iter, visible_beacons.end());
        EXPECT_EQ(iter->id, beacon.id);
        EXPECT_EQ(iter->pos_in_local, beacon.pos_in_local);
    }
}

TEST(BeaconTest, blinking_beacon_test) {
    using namespace std::literals::chrono_literals;
    // Setup
    const WorldMapConfig config = {
        .fixed_beacons = {},
        .blinking_beacons = {.beacons = {{.id = 1, .pos_in_local = {2.0, 3.0}},
                                         {.id = 4, .pos_in_local = {5.0, 6.0}},
                                         {.id = 7, .pos_in_local = {8.0, 9.0}}},
                             .beacon_appear_rate_hz = 1.0,
                             .beacon_disappear_rate_hz = 0.5},
        .correlated_beacons = {},
        .obstacles = {},
    };

    WorldMap map(config);
    constexpr time::RobotTimestamp MAX_TIME = time::RobotTimestamp() + 20s;
    constexpr time::RobotTimestamp::duration DT = 10ms;

    // Action
    for (time::RobotTimestamp t = time::RobotTimestamp(); t < MAX_TIME; t += DT) {
        map.update(t);
        const auto beacons = map.visible_beacons(t);
    }
}

TEST(BeaconTest, correlated_beacons_without_configuration) {
    // Setup
    const BeaconPotential potential =
        create_correlated_beacons({.p_beacon = 0.7, .p_no_beacons = 1e-6, .members = {1, 4, 7}});
    const WorldMapConfig config = {
        .fixed_beacons = {},
        .blinking_beacons = {},
        .correlated_beacons =
            {
                .beacons = {{.id = 1, .pos_in_local = {2.0, 3.0}},
                            {.id = 4, .pos_in_local = {5.0, 6.0}},
                            {.id = 7, .pos_in_local = {8.0, 9.0}}},
                .potential = potential,
                .configuration = {},
            },
        .obstacles = {},
    };

    // Action
    WorldMap map(config);

    // Verification
    // It is exceedingly unlikely that no beacons would be present
    EXPECT_GT(map.visible_beacons(time::RobotTimestamp()).size(), 0);
}

TEST(BeaconTest, correlated_beacons_with_configuration) {
    // Setup
    const BeaconPotential potential =
        create_correlated_beacons({.p_beacon = 0.7, .p_no_beacons = 0.25, .members = {1, 4, 7}});
    const WorldMapConfig config = {
        .fixed_beacons = {},
        .blinking_beacons = {},
        .correlated_beacons =
            {
                .beacons = {{.id = 1, .pos_in_local = {2.0, 3.0}},
                            {.id = 4, .pos_in_local = {5.0, 6.0}},
                            {.id = 7, .pos_in_local = {8.0, 9.0}}},
                .potential = potential,
                .configuration = {{false, true, true}},
            },
        .obstacles = {},
    };

    // Action
    WorldMap map(config);

    // Verification
    const auto visible_beacons = map.visible_beacons(time::RobotTimestamp());
    EXPECT_EQ(visible_beacons.size(), 2);
    EXPECT_EQ(visible_beacons.at(0).id, 4);
    EXPECT_EQ(visible_beacons.at(1).id, 7);
}
}  // namespace robot::experimental::beacon_sim
