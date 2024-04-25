
#include "experimental/beacon_sim/anticorrelated_beacon_potential_to_proto.hh"

namespace robot::experimental::beacon_sim {
namespace proto {
beacon_sim::AnticorrelatedBeaconPotential unpack_from(const AnticorrelatedBeaconPotential &in) {
    beacon_sim::AnticorrelatedBeaconPotential out;
    out.members.reserve(in.members().size());
    for (const auto member : in.members()) {
        out.members.push_back(member);
    }
    return out;
}

void pack_into(const beacon_sim::AnticorrelatedBeaconPotential &in,
               AnticorrelatedBeaconPotential *out) {
    out->mutable_members()->Clear();
    for (const auto member : in.members) {
        out->mutable_members()->Add(member);
    }
}

}  // namespace proto

void pack_into_potential(const AnticorrelatedBeaconPotential &in, proto::BeaconPotential *out) {
    pack_into(in, out->mutable_anticorrelated_potential());
}
}  // namespace robot::experimental::beacon_sim
