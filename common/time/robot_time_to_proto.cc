
#include "common/time/robot_time_to_proto.hh"

namespace robot::time::proto {
void pack_into(const time::RobotTimestamp &in, RobotTimestamp *out) {
    out->set_ticks_since_epoch(in.time_since_epoch().count());
}

time::RobotTimestamp unpack_from(const RobotTimestamp &in) {
    return time::RobotTimestamp() + time::RobotTimestamp::duration(in.ticks_since_epoch());
}

void pack_into(const time::RobotTimestamp::duration &in, RobotTimestampDuration *out) {
    out->set_count(in.count());
}

time::RobotTimestamp::duration unpack_from(const RobotTimestampDuration &in) {
    return time::RobotTimestamp::duration(in.count());
}
}  // namespace robot::time::proto
