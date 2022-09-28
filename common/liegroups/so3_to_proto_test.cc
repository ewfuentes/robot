
#include "common/liegroups/so3_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::liegroups {
TEST(SO3ToProtoTest, pack_unpack) {
    // Setup
    const liegroups::SO3 in = liegroups::SO3::exp({1.0, 2.0, 3.0});

    // Action
    liegroups::proto::SO3 proto;
    pack_into(in, &proto);
    const liegroups::SO3 out = unpack_from(proto);

    // Verification
    EXPECT_EQ((in.log() - out.log()).norm(), 0.0);
}

}  // namespace robot::liegroups
