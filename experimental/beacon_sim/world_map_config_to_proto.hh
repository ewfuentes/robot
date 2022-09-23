
#pragma once

#include "experimental/beacon_sim/world_map.hh"
#include "experimental/beacon_sim/world_map_config.pb.h"

namespace robot::experimental::beacon_sim::proto {
void pack_into(const beacon_sim::Beacon &in, Beacon *out);
beacon_sim::Beacon unpack_from(const Beacon &in);

void pack_into(const beacon_sim::FixedBeaconsConfig &in, FixedBeaconsConfig *out);
beacon_sim::FixedBeaconsConfig unpack_from(const FixedBeaconsConfig &in);

void pack_into(const beacon_sim::BlinkingBeaconsConfig &in, BlinkingBeaconsConfig *out);
beacon_sim::BlinkingBeaconsConfig unpack_from(const BlinkingBeaconsConfig &in);

void pack_into(const beacon_sim::WorldMapConfig &in, WorldMapConfig *out);
beacon_sim::WorldMapConfig unpack_from(const WorldMapConfig &in);
}  // namespace robot::experimental::beacon_sim::proto
