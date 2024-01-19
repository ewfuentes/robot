
#pragma once

#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/beacon_potential.pb.h"

namespace robot::experimental::beacon_sim {
namespace proto {
beacon_sim::BeaconPotential unpack_from(const BeaconPotential &in);
void pack_into(const beacon_sim::BeaconPotential &in, BeaconPotential *out);
beacon_sim::CombinedPotential unpack_from(const CombinedPotential &in);
void pack_into(const beacon_sim::CombinedPotential &in, CombinedPotential *out);
}  // namespace proto

void pack_into_potential(const CombinedPotential &in, BeaconPotential *out);

}  // namespace robot::experimental::beacon_sim
