
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(MappedLandmarksToProtoTest, pack_unpack) {
    // Setup
    const MappedLandmarks in = {
        .beacon_ids = {123, 234},
        .beacon_in_local = {{3.0, 4.0}, {5.0, 6.0}},
        .cov_in_local = Eigen::MatrixXd{{5.0, 6.0, 7.0, 8.0},
                                        {6.0, 7.0, 8.0, 9.0},
                                        {7.0, 8.0, 9.0, 0.0},
                                        {8.0, 9.0, 0.0, 1.0}},
    };

    // Action
    proto::MappedLandmarks proto;
    pack_into(in, &proto);
    const MappedLandmarks out = unpack_from(proto);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_EQ(in.beacon_ids.size(), out.beacon_ids.size());
    EXPECT_EQ(in.beacon_in_local.size(), out.beacon_in_local.size());
    for (int i = 0; i < static_cast<int>(in.beacon_ids.size()); i++) {
        EXPECT_EQ(in.beacon_ids.at(i), out.beacon_ids.at(i));
        EXPECT_EQ(in.beacon_in_local.at(i), out.beacon_in_local.at(i));
    }
    EXPECT_NEAR((in.cov_in_local - out.cov_in_local).norm(), 0.0, TOL);
}
}  // namespace robot::experimental::beacon_sim
