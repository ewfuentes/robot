
#include "experimental/beacon_sim/beacon_potential_to_proto.hh"

#include <algorithm>
#include <iterator>

#include "common/check.hh"
#include "experimental/beacon_sim/conditioned_potential_to_proto.hh"
#include "experimental/beacon_sim/correlated_beacon_potential_to_proto.hh"
#include "experimental/beacon_sim/precision_matrix_potential_to_proto.hh"

namespace robot::experimental::beacon_sim {
namespace proto {

beacon_sim::BeaconPotential unpack_from(const BeaconPotential &in) {
    switch (in.potential_oneof_case()) {
        case BeaconPotential::POTENTIAL_ONEOF_NOT_SET:
            return beacon_sim::BeaconPotential();
        case BeaconPotential::kCombinedPotential:
            return unpack_from(in.combined_potential());
        case BeaconPotential::kPrecisionMatrixPotential:
            return unpack_from(in.precision_matrix_potential());
        case BeaconPotential::kCorrelatedBeaconPotential:
            return unpack_from(in.correlated_beacon_potential());
        case BeaconPotential::kConditionedPotential:
            return unpack_from(in.conditioned_potential());
    }
    CHECK(false, "Unhandled potential type");
    return beacon_sim::BeaconPotential();
}

void pack_into(const beacon_sim::BeaconPotential &in, BeaconPotential *out) {
    out->clear_potential_oneof();
    if (in.impl_) {
        in.impl_->pack_into_(out);
    }
}

beacon_sim::CombinedPotential unpack_from(const CombinedPotential &in) {
    std::vector<beacon_sim::BeaconPotential> pots;
    std::transform(in.potentials().begin(), in.potentials().end(), std::back_inserter(pots),
                   [](const BeaconPotential &proto_pot) -> beacon_sim::BeaconPotential {
                       return unpack_from(proto_pot);
                   });
    return beacon_sim::CombinedPotential{
        .pots = std::move(pots),
    };
}

void pack_into(const beacon_sim::CombinedPotential &in, CombinedPotential *out) {
    for (const auto &pot : in.pots) {
        pack_into(pot, out->add_potentials());
    }
}

}  // namespace proto

void pack_into_potential(const CombinedPotential &in, proto::BeaconPotential *out) {
    pack_into(in, out->mutable_combined_potential());
}
}  // namespace robot::experimental::beacon_sim
