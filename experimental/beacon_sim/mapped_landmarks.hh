
#include "experimental/beacon_sim/world_map.hh"

#include "Eigen/Dense"

namespace robot::experimental::beacon_sim {

struct MappedLandmarks {
    struct LandmarkBelief {
        Beacon beacon;
        Eigen::Matrix2d cov_in_local;
    };
    std::vector<LandmarkBelief> landmarks;
};
}  // namespace robot::experimental::beacon_sim
