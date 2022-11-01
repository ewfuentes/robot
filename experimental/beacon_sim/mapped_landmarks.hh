
#pragma once

#include "Eigen/Dense"
#include "experimental/beacon_sim/world_map.hh"

namespace robot::experimental::beacon_sim {

struct MappedLandmarks {
    std::vector<int> beacon_ids;
    std::vector<Eigen::Vector2d> beacon_in_local;
    Eigen::MatrixXd cov_in_local;
};
}  // namespace robot::experimental::beacon_sim
