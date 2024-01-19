
#include "experimental/beacon_sim/conditioned_potential_to_proto.hh"

#include "experimental/beacon_sim/beacon_potential.pb.h"

namespace robot::experimental::beacon_sim {
namespace proto {

beacon_sim::ConditionedPotential unpack_from(const ConditionedPotential &in) {
    return {
        .underlying_pot = unpack_from(in.underlying_potential()),
        .log_normalizer = in.log_normalizer(),
        .conditioned_members = std::unordered_map(in.beacon_presence_from_id().begin(),
                                                  in.beacon_presence_from_id().end()),
    };
}

void pack_into(const beacon_sim::ConditionedPotential &in, ConditionedPotential *out) {
    pack_into(in.underlying_pot, out->mutable_underlying_potential());
    out->set_log_normalizer(in.log_normalizer);

    auto &presence_from_id = *out->mutable_beacon_presence_from_id();
    presence_from_id.clear();
    presence_from_id.insert(in.conditioned_members.begin(), in.conditioned_members.end());
}

}  // namespace proto

void pack_into_potential(const ConditionedPotential &in, proto::BeaconPotential *out) {
    pack_into(in, out->mutable_conditioned_potential());
}
}  // namespace robot::experimental::beacon_sim
