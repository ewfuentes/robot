

#include "common/liegroups/se3_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::liegroups {
  TEST(SE3ToProtoTest, pack_unpack) {
    // Setup
    const liegroups::SE3 in = liegroups::SE3::exp({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    // Action
    proto::SE3 proto;
    pack_into(in, &proto);
    const liegroups::SE3 out = unpack_from(proto);

    // Verification
    EXPECT_EQ((in.log() - out.log()).norm(), 0.0);
  }
}  // namespace robot::liegroups
