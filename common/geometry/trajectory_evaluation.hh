#pragma once
#include <cstddef>
#include <vector>

#include "Eigen/Core"

namespace robot::geometry {
const std::vector<double> get_absolute_trajectory_error(const std::vector<Eigen::Vector3d> &traj1,
                                                        const std::vector<Eigen::Vector3d> &traj2) {
    assert(traj1.size() == traj2.size());
    std::vector<double> result;
    for (size_t i = 0; i < traj2.size(); i++) {
        result.push_back((traj2[i] - traj1[i]).norm());
    }
    return result;
}
}  // namespace robot::geometry