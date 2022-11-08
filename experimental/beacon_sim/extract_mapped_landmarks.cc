
#include "experimental/beacon_sim/extract_mapped_landmarks.hh"

namespace robot::experimental::beacon_sim {
MappedLandmarks extract_mapped_landmarks(const EkfSlamEstimate &est) {
    MappedLandmarks out;
    constexpr int ROBOT_DIM = 3;
    const int beacon_cov_dim = est.mean.rows() - ROBOT_DIM;
    out.cov_in_local = est.cov.bottomRightCorner(beacon_cov_dim, beacon_cov_dim);

    for (const int beacon_id : est.beacon_ids) {
        out.beacon_ids.push_back(beacon_id);
        out.beacon_in_local.push_back(est.beacon_in_local(beacon_id).value());
    }
    return out;
}
}  // namespace robot::experimental::beacon_sim
