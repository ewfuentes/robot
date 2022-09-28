
#pragma once

#include <vector>

#include "experimental/beacon_sim/beacon_observation.pb.h"
#include "experimental/beacon_sim/generate_observations.hh"

namespace robot::experimental::beacon_sim::proto {

void pack_into(const beacon_sim::BeaconObservation &in, BeaconObservation *out);
beacon_sim::BeaconObservation unpack_from(const BeaconObservation &out);

void pack_into(const std::vector<beacon_sim::BeaconObservation> &in, BeaconObservations *out);
std::vector<beacon_sim::BeaconObservation> unpack_from(const BeaconObservations &in);

}  // namespace robot::experimental::beacon_sim::proto
