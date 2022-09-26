
#pragma once

#include "common/time/robot_time.hh"
#include "common/time/robot_time.pb.h"

namespace robot::time::proto {
void pack_into(const time::RobotTimestamp &in, RobotTimestamp *out);
time::RobotTimestamp unpack_from(const RobotTimestamp &in);
}  // namespace robot::time::proto
