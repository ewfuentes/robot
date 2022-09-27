
#include "experimental/beacon_sim/ekf_slam_estimate_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(EkfSlamEstimateToProtoTest, pack_unpack) {
    // Setup
    const EkfSlamEstimate in = {
        .mean = Eigen::VectorXd{{1.0, 2.0, 3.0, 4.0}},
        .cov = Eigen::MatrixXd{{0.1, 0.2, 0.3}, {90.0, 80.0, 70.0}, {-100.0, -200.0, -300.0}},
        .beacon_ids = {123, 234, 345, 567},
    };

    // Action
    proto::EkfSlamEstimate proto;
    pack_into(in, &proto);
    const EkfSlamEstimate out = unpack_from(proto);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_NEAR((in.mean - out.mean).norm(), 0.0, TOL);
    EXPECT_NEAR((in.cov - out.cov).norm(), 0.0, TOL);
    EXPECT_EQ(in.beacon_ids.size(), out.beacon_ids.size());
    for (int i = 0; i < static_cast<int>(in.beacon_ids.size()); i++) {
        EXPECT_EQ(in.beacon_ids.at(i), out.beacon_ids.at(i));
    }
}
}  // namespace robot::experimental::beacon_sim
