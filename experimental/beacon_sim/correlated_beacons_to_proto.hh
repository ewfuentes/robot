
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/correlated_beacons.pb.h"

namespace robot::experimental::beacon_sim::proto {
void pack_into(const beacon_sim::BeaconPotential &in, BeaconPotential *out);
beacon_sim::BeaconPotential unpack_from(const BeaconPotential &in);
}  // namespace robot::experimental::beacon_sim::proto
