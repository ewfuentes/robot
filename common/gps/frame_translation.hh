#pragma once

#include "Eigen/Core"

namespace robot::gps {
Eigen::Vector3d lla_from_ecef(const Eigen::Vector3d& t_place_from_ECEF);
Eigen::Vector3d ecef_from_lla(const Eigen::Vector3d& gcs_coordinate);
}  // namespace robot::gps
