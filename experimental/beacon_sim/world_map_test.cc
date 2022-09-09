
#include "experimental/beacon_sim/world_map.hh"

#include <chrono>

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(BeaconTest, static_beacon_test) {
    // Setup
    const WorldMapOptions config = {
        .fixed_beacons = {.beacons = {{.id = 1, .pos_in_local = {2.0, 3.0}},
                                      {.id = 4, .pos_in_local = {5.0, 6.0}},
                                      {.id = 7, .pos_in_local = {8.0, 9.0}}}},
        .blinking_beacons = {},
    };

    // Action
    const WorldMap map(config, nullptr);
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
    const WorldMapOptions config = {
        .fixed_beacons = {},
        .blinking_beacons = {.beacons = {{.id = 1, .pos_in_local = {2.0, 3.0}},
                                         {.id = 4, .pos_in_local = {5.0, 6.0}},
                                         {.id = 7, .pos_in_local = {8.0, 9.0}}},
                             .beacon_appear_rate_hz = 1.0,
                             .beacon_disappear_rate_hz = 0.5},
    };

    WorldMap map(config, std::make_unique<std::mt19937>(0));
    constexpr time::RobotTimestamp MAX_TIME = time::RobotTimestamp() + 20s;
    constexpr time::RobotTimestamp::duration DT = 10ms;

    // Action
    for (time::RobotTimestamp t = time::RobotTimestamp(); t < MAX_TIME; t += DT) {
        map.update(t);
        const auto beacons = map.visible_beacons(t);
    }
}
}  // namespace robot::experimental::beacon_sim
