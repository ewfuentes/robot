
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(MappedLandmarksToProtoTest, pack_unpack) {
    // Setup
    const MappedLandmarks in = {
        .landmarks =
            {
                {.beacon = {.id = 123, .pos_in_local = {3.0, 4.0}},
                 .cov_in_local = Eigen::Matrix2d{{{1.0, 2.0}, {2.0, 1.0}}}},
                {.beacon = {.id = 234, .pos_in_local = {5.0, 6.0}},
                 .cov_in_local = Eigen::Matrix2d{{{3.0, 4.0}, {4.0, 3.0}}}},
            },
    };

    // Action
    proto::MappedLandmarks proto;
    pack_into(in, &proto);
    const MappedLandmarks out = unpack_from(proto);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_EQ(in.landmarks.size(), out.landmarks.size());
    for (int i = 0; i < static_cast<int>(in.landmarks.size()); i++) {
        const auto &in_landmark = in.landmarks.at(i);
        const auto &out_landmark = out.landmarks.at(i);
        EXPECT_EQ(in_landmark.beacon.id, out_landmark.beacon.id);
        EXPECT_NEAR((in_landmark.beacon.pos_in_local - out_landmark.beacon.pos_in_local).norm(),
                    0.0, TOL);
        EXPECT_NEAR((in_landmark.cov_in_local - out_landmark.cov_in_local).norm(), 0.0, TOL);
    }
}
}  // namespace robot::experimental::beacon_sim
