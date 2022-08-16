
#include "experimental/beacon_sim/robot.hh"

#include <cmath>

namespace experimental::beacon_sim {
void RobotState::move(const double distance_m) {
    pos_m_ += distance_m * Eigen::Vector2d{std::cos(heading_rad_), std::sin(heading_rad_)};
}

void RobotState::turn(const double angle_rad) { heading_rad_ += angle_rad; }
}  // namespace experimental::beacon_sim
