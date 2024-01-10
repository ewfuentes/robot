
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
}
}  // namespace robot::experimental::beacon_sim
