
#include "experimental/beacon_sim/world_map.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(BeaconTest, static_beacon_test) {
    // Setup
    time::set_default_time_provider(time::TimeProvider::SIM);
    const WorldMapOptions config = {
        .fixed_beacons = {.beacons = {{.id = 1, .pos_in_local = {2.0, 3.0}},
                                      {.id = 4, .pos_in_local = {5.0, 6.0}},
                                      {.id = 7, .pos_in_local = {8.0, 9.0}}}},
        .blinking_beacons = {},
    };

    // Action
    const WorldMap map(config, nullptr);
    const auto beacons = map.visible_beacons(time::current_robot_time());

    // Verification
    EXPECT_EQ(beacons.size(), config.fixed_beacons.beacons.size());
}
}  // namespace robot::experimental::beacon_sim
