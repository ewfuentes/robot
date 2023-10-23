
#pragma once

#include "planning/road_map.hh"
#include "planning/road_map.pb.h"

namespace robot::planning::proto {
void pack_into(const planning::RoadMap &in, RoadMap *out);
planning::RoadMap unpack_from(const RoadMap &in);
}  // namespace robot::planning::proto
