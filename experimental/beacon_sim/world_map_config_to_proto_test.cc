
#include "experimental/beacon_sim/world_map_config_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(WorldMapConfigToProtoTest, beacon_pack_unpack) {
    // Setup
    const Beacon beacon = {
        .id = 1,
        .pos_in_local = {2.3, 3.4},
    };

    // Action
    proto::Beacon proto;
    pack_into(beacon, &proto);
    const Beacon other = unpack_from(proto);

    // Verification
    EXPECT_EQ(beacon.id, other.id);
    EXPECT_EQ(beacon.pos_in_local.x(), other.pos_in_local.x());
    EXPECT_EQ(beacon.pos_in_local.y(), other.pos_in_local.y());
}

TEST(WorldMapConfigToProtoTest, fixed_beacons_config_pack_unpack) {
    // Setup
    const FixedBeaconsConfig config = {
        .beacons =
            {
                {
                    .id = 1,
                    .pos_in_local = {2.3, 3.4},
                },
                {
                    .id = 2,
                    .pos_in_local = {4.5, 5.6},
                },
            },
    };

    // Action
    proto::FixedBeaconsConfig proto;
    pack_into(config, &proto);
    const FixedBeaconsConfig other = unpack_from(proto);

    // Verification
    EXPECT_EQ(config.beacons.size(), other.beacons.size());
    for (int i = 0; i < static_cast<int>(config.beacons.size()); i++) {
        const auto &beacon = config.beacons.at(i);
        const auto &other_beacon = other.beacons.at(i);
        EXPECT_EQ(beacon.pos_in_local.x(), other_beacon.pos_in_local.x());
        EXPECT_EQ(beacon.pos_in_local.y(), other_beacon.pos_in_local.y());
    }
}

TEST(WorldMapConfigToProtoTest, blinking_beacons_config_pack_unpack) {
    // Setup
    const BlinkingBeaconsConfig config = {
        .beacons =
            {
                {
                    .id = 1,
                    .pos_in_local = {2.3, 3.4},
                },
                {
                    .id = 2,
                    .pos_in_local = {4.5, 5.6},
                },
            },
        .beacon_appear_rate_hz = 100.0,
        .beacon_disappear_rate_hz = -300.0,
    };

    // Action
    proto::BlinkingBeaconsConfig proto;
    pack_into(config, &proto);
    const BlinkingBeaconsConfig other = unpack_from(proto);

    // Verification
    EXPECT_EQ(config.beacons.size(), other.beacons.size());
    for (int i = 0; i < static_cast<int>(config.beacons.size()); i++) {
        const auto &beacon = config.beacons.at(i);
        const auto &other_beacon = other.beacons.at(i);
        EXPECT_EQ(beacon.pos_in_local.x(), other_beacon.pos_in_local.x());
        EXPECT_EQ(beacon.pos_in_local.y(), other_beacon.pos_in_local.y());
    }
    EXPECT_EQ(config.beacon_appear_rate_hz, other.beacon_appear_rate_hz);
    EXPECT_EQ(config.beacon_disappear_rate_hz, other.beacon_disappear_rate_hz);
}

TEST(WorldMapConfigToProtoTest, world_map_config_pack_unpack) {
    // Setup

    const WorldMapConfig config = {
        .fixed_beacons = {.beacons =
                              {
                                  {
                                      .id = 1,
                                      .pos_in_local = {2.3, 3.4},
                                  },
                              }},
        .blinking_beacons =
            {
                .beacons =
                    {
                        {
                            .id = 2,
                            .pos_in_local = {4.5, 5.6},
                        },
                    },
                .beacon_appear_rate_hz = 100.0,
                .beacon_disappear_rate_hz = -300.0,
            },
        .correlated_beacons =
            {
                .beacons =
                    {
                        {
                            .id = 3,
                            .pos_in_local = {7.8, 9.0},
                        },
                    },
                .potential = BeaconPotential(Eigen::MatrixXd::Zero(1, 1), 1.0, {3}),
            },
        // TODO Add obstacles to pack/unpack
        .obstacles = {},
    };

    // Action
    proto::WorldMapConfig proto;
    pack_into(config, &proto);
    const WorldMapConfig other = unpack_from(proto);

    // Verification
    // Assume that if the correct number of items were unpacked, the rest of it is correct
    EXPECT_EQ(config.fixed_beacons.beacons.size(), other.fixed_beacons.beacons.size());
    EXPECT_EQ(config.blinking_beacons.beacons.size(), other.blinking_beacons.beacons.size());
    EXPECT_EQ(config.correlated_beacons.beacons.size(), other.correlated_beacons.beacons.size());
    EXPECT_EQ(config.correlated_beacons.potential.members().size(),
              other.correlated_beacons.potential.members().size());
}

}  // namespace robot::experimental::beacon_sim
