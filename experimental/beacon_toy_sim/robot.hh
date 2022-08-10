
#pragma once

namespace experimental::beacon_toy_sim {
  struct RobotState {
    RobotState(const double pos_x_m, const double pos_y_m, const double heading_rad) : pos_x_m_(pos_x_m), pos_y_m_(pos_y_m), heading_rad_(heading_rad){}

    void move(const double distance_m);
    void turn(const double angle_rad);

    const double &pos_x_m() const { return pos_x_m_; }
    const double &pos_y_m() const { return pos_y_m_; }
    const double &heading_rad() const { return heading_rad_; }
  private:
    double pos_x_m_;
    double pos_y_m_;
    double heading_rad_;
  };
}
