
#include "experimental/beacon_sim/world_map.hh"

namespace robot::experimental::beacon_sim {
WorldMap::WorldMap(const WorldMapOptions &options) : beacons_(options.fixed_beacons.beacons) {}
}  // namespace robot::experimental::beacon_sim
