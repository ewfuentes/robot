
#include "common/liegroups/se2_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::liegroups {
TEST(SE2ToProtoTest, pack_unpack) {
    // Setup
    const liegroups::SE2 in = liegroups::SE2::exp({1.0, 2.0, 3.0});

    // Action
    proto::SE2 proto;
    pack_into(in, &proto);
    const liegroups::SE2 out = unpack_from(proto);

    // Verification
    EXPECT_EQ((in.log() - out.log()).norm(), 0.0);
}
}  // namespace robot::liegroups
