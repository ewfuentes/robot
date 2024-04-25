
#include "experimental/beacon_sim/anticorrelated_beacon_potential.hh"
#include "experimental/beacon_sim/anticorrelated_beacon_potential.pb.h"
#include "experimental/beacon_sim/beacon_potential.pb.h"

namespace robot::experimental::beacon_sim {
namespace proto {
beacon_sim::AnticorrelatedBeaconPotential unpack_from(const AnticorrelatedBeaconPotential &in);
void pack_into(const beacon_sim::AnticorrelatedBeaconPotential &in,
               AnticorrelatedBeaconPotential *out);
}  // namespace proto

void pack_into_potential(const AnticorrelatedBeaconPotential &in, proto::BeaconPotential *);
}  // namespace robot::experimental::beacon_sim
