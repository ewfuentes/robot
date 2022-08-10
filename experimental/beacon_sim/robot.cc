
#include "experimental/beacon_sim/robot.hh"

#include <cmath>

namespace experimental::beacon_sim {
  void RobotState::move(const double distance_m) {
    pos_x_m_ = pos_x_m_ + distance_m * std::cos(heading_rad_);
    pos_y_m_ = pos_y_m_ + distance_m * std::sin(heading_rad_);
  }

  void RobotState::turn(const double angle_rad) {
    heading_rad_ += angle_rad;
  }
}
