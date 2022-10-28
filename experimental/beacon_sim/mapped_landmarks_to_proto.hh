
#include "experimental/beacon_sim/mapped_landmarks.hh"
#include "experimental/beacon_sim/mapped_landmarks.pb.h"

namespace robot::experimental::beacon_sim::proto {
void pack_into(const beacon_sim::MappedLandmarks &in, MappedLandmarks *out);
beacon_sim::MappedLandmarks unpack_from(const MappedLandmarks &in);
}  // namespace robot::experimental::beacon_sim::proto
