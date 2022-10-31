
#include "experimental/beacon_sim/extract_mapped_landmarks.hh"

namespace robot::experimental::beacon_sim {
MappedLandmarks extract_mapped_landmarks(const EkfSlamEstimate &est) {
    MappedLandmarks out;

    for (const int beacon_id : est.beacon_ids) {
        out.landmarks.push_back({
            .beacon =
                {
                    .id = beacon_id,
                    .pos_in_local = est.beacon_in_local(beacon_id).value(),
                },
            .cov_in_local = est.beacon_cov(beacon_id).value(),
        });
    }
    return out;
}
}  // namespace robot::experimental::beacon_sim
