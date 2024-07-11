
#pragma once

#include "experimental/beacon_sim/beacon_potential.pb.h"
#include "experimental/beacon_sim/correlated_beacon_potential.hh"
#include "experimental/beacon_sim/correlated_beacon_potential.pb.h"

namespace robot::experimental::beacon_sim {
namespace proto {
beacon_sim::CorrelatedBeaconPotential::ConditioningBlock unpack_from(
    const CorrelatedConditioningBlock &in);
void pack_into(const beacon_sim::CorrelatedBeaconPotential::ConditioningBlock &in,
               CorrelatedConditioningBlock *out);

beacon_sim::CorrelatedBeaconPotential unpack_from(const CorrelatedBeaconPotential &in);
void pack_into(const beacon_sim::CorrelatedBeaconPotential &in, CorrelatedBeaconPotential *out);
}  // namespace proto
void pack_into_potential(const CorrelatedBeaconPotential &in, proto::BeaconPotential *out);
}  // namespace robot::experimental::beacon_sim
