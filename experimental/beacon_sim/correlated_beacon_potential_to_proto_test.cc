
#include "experimental/beacon_sim/correlated_beacon_potential_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(CorrelatedBeaconPotentialToProtoTest, pack_unpack) {
    // Setup
    const CorrelatedBeaconPotential pot = {
        .p_present = 0.4, .p_beacon_given_present = 0.1, .members = {1, 2, 3}};

    // Action
    proto::CorrelatedBeaconPotential proto;
    proto::pack_into(pot, &proto);
    const CorrelatedBeaconPotential unpacked = unpack_from(proto);

    // Verification
    EXPECT_EQ(pot.p_present, unpacked.p_present);
    EXPECT_EQ(pot.p_beacon_given_present, unpacked.p_beacon_given_present);
    EXPECT_EQ(pot.members.size(), unpacked.members.size());
    for (int i = 0; i < static_cast<int>(pot.members.size()); i++) {
        EXPECT_EQ(pot.members.at(i), unpacked.members.at(i));
    }
    EXPECT_FALSE(pot.conditioning.has_value());
}

TEST(CorrelatedBeaconPotentialToProtoTest, pack_unpack_conditioned) {
    // Setup
    const CorrelatedBeaconPotential pot = {
        .p_present = 0.4,
        .p_beacon_given_present = 0.1,
        .members = {1, 2, 3},
        .conditioning = {{.conditioned_members = {{1, false}, {3, true}}}}};

    // Action
    proto::CorrelatedBeaconPotential proto;
    proto::pack_into(pot, &proto);
    const CorrelatedBeaconPotential unpacked = unpack_from(proto);

    // Verification
    EXPECT_EQ(pot.p_present, unpacked.p_present);
    EXPECT_EQ(pot.p_beacon_given_present, unpacked.p_beacon_given_present);
    EXPECT_EQ(pot.members.size(), unpacked.members.size());
    for (int i = 0; i < static_cast<int>(pot.members.size()); i++) {
        EXPECT_EQ(pot.members.at(i), unpacked.members.at(i));
    }

    EXPECT_EQ(pot.conditioning->conditioned_members.size(),
              unpacked.conditioning->conditioned_members.size());
    for (const auto &[id, value] : pot.conditioning->conditioned_members) {
        EXPECT_TRUE(unpacked.conditioning->conditioned_members.contains(id));
        EXPECT_EQ(unpacked.conditioning->conditioned_members.at(id), value);
    }
}
}  // namespace robot::experimental::beacon_sim
