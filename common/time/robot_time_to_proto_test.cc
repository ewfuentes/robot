
#include "gtest/gtest.h"

#include "common/time/robot_time_to_proto.hh"

namespace robot::time {
  TEST(RobotTimeToProtoTest, pack_unpack) {
    // Setup
    const time::RobotTimestamp in = RobotTimestamp() + std::chrono::milliseconds(123456);

    // Action
    time::proto::RobotTimestamp proto;
    pack_into(in, &proto);
    const time::RobotTimestamp out = unpack_from(proto);

    // Verification
    EXPECT_EQ(in, out);

  }
}
