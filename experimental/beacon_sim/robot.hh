
#pragma once

#include "common/liegroups/se2.hh"

namespace robot::experimental::beacon_sim {
struct RobotState {
    RobotState(const double pos_x_m, const double pos_y_m, const double heading_rad)
        : local_from_robot_(liegroups::SO2(heading_rad), Eigen::Vector2d{pos_x_m, pos_y_m}) {}
    explicit RobotState(const liegroups::SE2 &local_from_robot)
        : local_from_robot_(local_from_robot) {}

    void move(const double distance_m);
    void turn(const double angle_rad);

    double pos_x_m() const { return local_from_robot_.translation().x(); }
    double pos_y_m() const { return local_from_robot_.translation().y(); }
    double heading_rad() const { return local_from_robot_.so2().log(); }
    const liegroups::SE2 &local_from_robot() const { return local_from_robot_; }

   private:
    liegroups::SE2 local_from_robot_;
};
}  // namespace robot::experimental::beacon_sim
