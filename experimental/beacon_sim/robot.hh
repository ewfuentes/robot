
#pragma once

#include "sophus/se2.hpp"

namespace experimental::beacon_sim {
struct RobotState {
    RobotState(const double pos_x_m, const double pos_y_m, const double heading_rad)
        : local_from_robot_(Sophus::SO2(heading_rad), Eigen::Vector2d{pos_x_m, pos_y_m}) {}

    void move(const double distance_m);
    void turn(const double angle_rad);

    double pos_x_m() const { return local_from_robot_.translation().x(); }
    double pos_y_m() const { return local_from_robot_.translation().y(); }
    double heading_rad() const { return local_from_robot_.so2().log(); }
    const Sophus::SE2d &local_from_robot() const { return local_from_robot_; }

   private:
    Sophus::SE2d local_from_robot_;
};
}  // namespace experimental::beacon_sim
