
#pragma once

#include "Eigen/Dense"
#include "experimental/beacon_sim/world_map.hh"

namespace robot::experimental::beacon_sim {

struct MappedLandmarks {
    struct LandmarkBelief {
        Beacon beacon;
        Eigen::Matrix2d cov_in_local;
    };
    std::vector<LandmarkBelief> landmarks;
};
}  // namespace robot::experimental::beacon_sim
