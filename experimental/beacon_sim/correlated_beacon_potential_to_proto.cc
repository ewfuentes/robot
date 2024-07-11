
#include "experimental/beacon_sim/correlated_beacon_potential_to_proto.hh"

#include "experimental/beacon_sim/correlated_beacon_potential.hh"

namespace robot::experimental::beacon_sim {
namespace proto {

beacon_sim::CorrelatedBeaconPotential::ConditioningBlock unpack_from(
    const CorrelatedConditioningBlock &in) {
    return {.conditioned_members = std::unordered_map(in.conditioned_members().begin(),
                                                      in.conditioned_members().end())};
}

void pack_into(const beacon_sim::CorrelatedBeaconPotential::ConditioningBlock &in,
               CorrelatedConditioningBlock *out) {
    out->mutable_conditioned_members()->clear();
    out->mutable_conditioned_members()->insert(in.conditioned_members.begin(),
                                               in.conditioned_members.end());
}

beacon_sim::CorrelatedBeaconPotential unpack_from(const CorrelatedBeaconPotential &in) {
    std::vector<int> members(in.members().begin(), in.members().end());
    return beacon_sim::CorrelatedBeaconPotential{
        .p_present = in.p_present(),
        .p_beacon_given_present = in.p_beacon_given_present(),
        .members = std::move(members),
        .conditioning = in.has_conditioning() ? std::make_optional(unpack_from(in.conditioning()))
                                              : std::nullopt};
}

void pack_into(const beacon_sim::CorrelatedBeaconPotential &in, CorrelatedBeaconPotential *out) {
    out->set_p_present(in.p_present);
    out->set_p_beacon_given_present(in.p_beacon_given_present);
    for (const auto beacon_id : in.members) {
        out->add_members(beacon_id);
    }
    if (in.conditioning.has_value()) {
        pack_into(in.conditioning.value(), out->mutable_conditioning());
    }
}
}  // namespace proto

void pack_into_potential(const CorrelatedBeaconPotential &in, proto::BeaconPotential *out) {
    pack_into(in, out->mutable_correlated_beacon_potential());
}

}  // namespace robot::experimental::beacon_sim
