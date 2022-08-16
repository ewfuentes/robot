
#pragma once

#include "Eigen/Core"

namespace experimental::beacon_sim {
struct RobotState {
    RobotState(const double pos_x_m, const double pos_y_m, const double heading_rad)
        : pos_m_{pos_x_m, pos_y_m}, heading_rad_(heading_rad) {}

    void move(const double distance_m);
    void turn(const double angle_rad);

    const double &pos_x_m() const { return pos_m_(0); }
    const double &pos_y_m() const { return pos_m_(1); }
    const double &heading_rad() const { return heading_rad_; }

   private:
    Eigen::Vector2d pos_m_;
    double heading_rad_;
};
}  // namespace experimental::beacon_sim
