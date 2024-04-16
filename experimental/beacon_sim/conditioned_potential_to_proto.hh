
#pragma once

#include "beacon_potential.hh"
#include "experimental/beacon_sim/beacon_potential.pb.h"

namespace robot::experimental::beacon_sim {
namespace proto {
class BeaconPotential;

beacon_sim::ConditionedPotential unpack_from(const ConditionedPotential &in);
void pack_into(const beacon_sim::ConditionedPotential &in, ConditionedPotential *out);
}  // namespace proto

void pack_into_potential(const ConditionedPotential &in, proto::BeaconPotential *out);

}  // namespace robot::experimental::beacon_sim
