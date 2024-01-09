
#include "experimental/beacon_sim/world_map_config_to_proto.hh"

#include "experimental/beacon_sim/precision_matrix_potential.hh"
#include "experimental/beacon_sim/world_map.hh"
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

TEST(WorldMapConfigToProtoTest, correlated_beacons_config_without_configuration_pack_unpack) {
    // Setup
    const CorrelatedBeaconsConfig config = {
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
        .potential =
            PrecisionMatrixPotential{
                .precision = Eigen::Matrix2d::Identity(), .log_normalizer = 7.8, .members = {1, 2}},
        .configuration = {},
    };

    // Action
    proto::CorrelatedBeaconsConfig proto;
    proto::pack_into(config, &proto);
    const CorrelatedBeaconsConfig other = unpack_from(proto);

    // Verification
    EXPECT_EQ(config.beacons.size(), other.beacons.size());
    for (int i = 0; i < static_cast<int>(config.beacons.size()); i++) {
        const auto &beacon = config.beacons.at(i);
        const auto &other_beacon = other.beacons.at(i);
        EXPECT_EQ(beacon.pos_in_local.x(), other_beacon.pos_in_local.x());
        EXPECT_EQ(beacon.pos_in_local.y(), other_beacon.pos_in_local.y());
    }
    EXPECT_EQ(config.potential.members().size(), other.potential.members().size());
    for (int i = 0; i < static_cast<int>(config.potential.members().size()); i++) {
        EXPECT_EQ(config.potential.members().at(i), other.potential.members().at(i));
    }
    EXPECT_EQ(config.configuration.has_value(), other.configuration.has_value());
}

TEST(WorldMapConfigToProtoTest, correlated_beacons_config_with_configuration_pack_unpack) {
    // Setup
    const CorrelatedBeaconsConfig config = {
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
        .potential =
            PrecisionMatrixPotential{
                .precision = Eigen::Matrix2d::Identity(), .log_normalizer = 7.8, .members = {1, 2}},
        .configuration = {{false, true}},
    };

    // Action
    proto::CorrelatedBeaconsConfig proto;
    proto::pack_into(config, &proto);
    const CorrelatedBeaconsConfig other = unpack_from(proto);

    // Verification
    EXPECT_EQ(config.beacons.size(), other.beacons.size());
    for (int i = 0; i < static_cast<int>(config.beacons.size()); i++) {
        const auto &beacon = config.beacons.at(i);
        const auto &other_beacon = other.beacons.at(i);
        EXPECT_EQ(beacon.pos_in_local.x(), other_beacon.pos_in_local.x());
        EXPECT_EQ(beacon.pos_in_local.y(), other_beacon.pos_in_local.y());
    }
    EXPECT_EQ(config.potential.members().size(), other.potential.members().size());
    for (int i = 0; i < static_cast<int>(config.potential.members().size()); i++) {
        EXPECT_EQ(config.potential.members().at(i), other.potential.members().at(i));
    }
    EXPECT_EQ(config.configuration.has_value(), other.configuration.has_value());
    for (int i = 0; i < static_cast<int>(config.configuration->size()); i++) {
        EXPECT_EQ(config.configuration->at(i), other.configuration->at(i));
    }
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
                .potential = PrecisionMatrixPotential{.precision = Eigen::MatrixXd::Zero(1, 1),
                                                      .log_normalizer = 1.0,
                                                      .members = {3}},
                .configuration = {},
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
    EXPECT_FALSE(other.correlated_beacons.configuration.has_value());
}

}  // namespace robot::experimental::beacon_sim
