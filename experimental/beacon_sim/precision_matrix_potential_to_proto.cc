
#include "experimental/beacon_sim/precision_matrix_potential_to_proto.hh"

#include "common/math/matrix_to_proto.hh"
#include "experimental/beacon_sim/precision_matrix_potential.hh"

namespace robot::experimental::beacon_sim {

void pack_into_potential(const PrecisionMatrixPotential &in, proto::BeaconPotential *out) {
    pack_into(in, out->mutable_precision_matrix_potential());
}

namespace proto {
void pack_into(const beacon_sim::PrecisionMatrixPotential &in, PrecisionMatrixPotential *out) {
    pack_into(in.precision, out->mutable_precision());
    out->set_log_normalizer(in.log_normalizer);
    for (const auto &member : in.members) {
        out->add_members(member);
    }
}

beacon_sim::PrecisionMatrixPotential unpack_from(const PrecisionMatrixPotential &in) {
    return beacon_sim::PrecisionMatrixPotential{
        .precision = unpack_from<Eigen::MatrixXd>(in.precision()),
        .log_normalizer = in.log_normalizer(),
        .members = std::vector<int>(in.members().begin(), in.members().end())};
}
}  // namespace proto
}  // namespace robot::experimental::beacon_sim
