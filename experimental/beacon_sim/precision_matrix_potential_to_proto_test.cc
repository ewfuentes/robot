
#include "experimental/beacon_sim/precision_matrix_potential_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(CorrelatedBeaconsToProtoTest, pack_unpack) {
    // Setup
    const PrecisionMatrixPotential pot{
        .precision = Eigen::Matrix2d::Identity(), .log_normalizer = 123.0, .members = {45, 50}};
    // Action
    proto::PrecisionMatrixPotential msg;
    pack_into(pot, &msg);
    const PrecisionMatrixPotential unpacked = unpack_from(msg);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_NEAR((unpacked.precision - pot.precision).norm(), 0.0, TOL);
    EXPECT_NEAR(unpacked.log_normalizer, pot.log_normalizer, TOL);
    ASSERT_EQ(unpacked.members.size(), pot.members.size());
    EXPECT_EQ(unpacked.members.at(0), pot.members.at(0));
    EXPECT_EQ(unpacked.members.at(1), pot.members.at(1));
}
}  // namespace robot::experimental::beacon_sim
