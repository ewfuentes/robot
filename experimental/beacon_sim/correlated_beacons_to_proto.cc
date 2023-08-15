
#include "experimental/beacon_sim/correlated_beacons_to_proto.hh"

#include "common/math/matrix_to_proto.hh"

namespace robot::experimental::beacon_sim::proto {
void pack_into(const beacon_sim::BeaconPotential &in, BeaconPotential *out) {
    pack_into(in.precision(), out->mutable_precision());
    out->set_log_normalizer(in.log_normalizer());
    for (const auto &member : in.members()) {
        out->add_members(member);
    }
}

beacon_sim::BeaconPotential unpack_from(const BeaconPotential &in) {
    return beacon_sim::BeaconPotential(unpack_from<Eigen::MatrixXd>(in.precision()),
                                       in.log_normalizer(),
                                       std::vector<int>(in.members().begin(), in.members().end()));
}
}  // namespace robot::experimental::beacon_sim::proto
