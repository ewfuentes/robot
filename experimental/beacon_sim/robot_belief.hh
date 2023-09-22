
#pragma once

#include "Eigen/Core"

#include "common/liegroups/se2.hh"

namespace robot::experimental::beacon_sim {

struct RobotBelief {
    liegroups::SE2 local_from_robot;
    Eigen::Matrix3d cov_in_robot;
};
}

