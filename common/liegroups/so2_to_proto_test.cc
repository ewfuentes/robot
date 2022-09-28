
#include "common/liegroups/so2_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::liegroups {

TEST(SO2ToProtoTest, pack_unpack) {
    // Setup
    constexpr double THETA = 1.23;
    const liegroups::SO2 in(THETA);

    // Action
    proto::SO2 proto;
    pack_into(in, &proto);
    const liegroups::SO2 out = unpack_from(proto);

    // Verification
    EXPECT_EQ(in.log(), out.log());
}
}  // namespace robot::liegroups
