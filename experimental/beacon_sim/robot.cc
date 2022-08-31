
#include "experimental/beacon_sim/robot.hh"

#include <cmath>

namespace robot::experimental::beacon_sim {
void RobotState::move(const double distance_m) {
    const auto old_robot_from_new_robot = liegroups::SE2::transX(distance_m);
    local_from_robot_ = local_from_robot_ * old_robot_from_new_robot;
}

void RobotState::turn(const double angle_rad) {
    const auto old_robot_from_new_robot = liegroups::SE2::rot(angle_rad);
    local_from_robot_ = local_from_robot_ * old_robot_from_new_robot;
}
}  // namespace robot::experimental::beacon_sim
