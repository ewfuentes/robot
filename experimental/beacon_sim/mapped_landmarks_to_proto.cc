
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"

#include "experimental/beacon_sim/world_map_config_to_proto.hh"
#include "common/math/matrix_to_proto.hh"

namespace robot::experimental::beacon_sim::proto {
void pack_into(const beacon_sim::MappedLandmarks &in, MappedLandmarks *out) {
    for (const auto &landmark : in.landmarks) {
        auto proto_ptr = out->add_landmarks();
        pack_into(landmark.beacon, proto_ptr->mutable_beacon());
        pack_into(landmark.cov_in_local, proto_ptr->mutable_cov_in_local());
    }
}

beacon_sim::MappedLandmarks unpack_from(const MappedLandmarks &in) {
    beacon_sim::MappedLandmarks out;
    for (const auto &landmark : in.landmarks()) {
        out.landmarks.push_back({
            .beacon = unpack_from(landmark.beacon()),
            .cov_in_local = unpack_from<Eigen::Matrix2d>(landmark.cov_in_local()),
        });
    }
    return out;
}
}  // namespace robot::experimental::beacon_sim::proto
