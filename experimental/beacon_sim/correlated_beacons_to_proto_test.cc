
#include "experimental/beacon_sim/correlated_beacons_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(CorrelatedBeaconsToProtoTest, pack_unpack) {
    // Setup
    const BeaconPotential pot(Eigen::Matrix2d::Identity(), 123.0, {45, 50});
    // Action
    proto::BeaconPotential msg;
    pack_into(pot, &msg);
    const BeaconPotential unpacked = unpack_from(msg);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_NEAR((unpacked.covariance() - pot.covariance()).norm(), 0.0, TOL);
    EXPECT_NEAR(unpacked.bias(), pot.bias(), TOL);
    ASSERT_EQ(unpacked.members().size(), pot.members().size());
    EXPECT_EQ(unpacked.members().at(0), pot.members().at(0));
    EXPECT_EQ(unpacked.members().at(1), pot.members().at(1));
}
}  // namespace robot::experimental::beacon_sim
