
#include "common/time/robot_time_to_proto.hh"

#include "gtest/gtest.h"

namespace robot::time {
TEST(RobotTimeToProtoTest, pack_unpack_timestamp) {
    // Setup
    const time::RobotTimestamp in = RobotTimestamp() + std::chrono::milliseconds(123456);

    // Action
    time::proto::RobotTimestamp proto;
    pack_into(in, &proto);
    const time::RobotTimestamp out = unpack_from(proto);

    // Verification
    EXPECT_EQ(in, out);
}

TEST(RobotTimeToProtoTest, pack_unpack_duration) {
    // Setup
    const time::RobotTimestamp::duration in = std::chrono::milliseconds(123456);

    // Action
    time::proto::RobotTimestampDuration proto;
    pack_into(in, &proto);
    const time::RobotTimestamp::duration out = unpack_from(proto);

    // Verification
    EXPECT_EQ(in, out);
}
}  // namespace robot::time
