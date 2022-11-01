
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"

#include "common/math/matrix_to_proto.hh"
#include "experimental/beacon_sim/world_map_config_to_proto.hh"

namespace robot::experimental::beacon_sim::proto {
void pack_into(const beacon_sim::MappedLandmarks &in, MappedLandmarks *out) {
    for (const int id : in.beacon_ids) {
        out->add_beacon_ids(id);
    }

    for (const Eigen::Vector2d &beacon_in_local : in.beacon_in_local) {
        pack_into(beacon_in_local, out->add_beacon_in_local());
    }

    pack_into(in.cov_in_local, out->mutable_cov_in_local());
}

beacon_sim::MappedLandmarks unpack_from(const MappedLandmarks &in) {
    beacon_sim::MappedLandmarks out;
    for (const int id : in.beacon_ids()) {
        out.beacon_ids.push_back(id);
    }
    for (const auto &beacon_in_local : in.beacon_in_local()) {
        out.beacon_in_local.push_back(unpack_from<Eigen::Vector2d>(beacon_in_local));
    }
    out.cov_in_local = unpack_from<Eigen::MatrixXd>(in.cov_in_local());
    return out;
}
}  // namespace robot::experimental::beacon_sim::proto
