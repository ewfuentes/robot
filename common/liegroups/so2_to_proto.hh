#pragma once

#include "common/liegroups/so2.hh"
#include "common/liegroups/so2.pb.h"

namespace robot::liegroups::proto {

void pack_into(const liegroups::SO2 &in, SO2 *out);
liegroups::SO2 unpack_from(const SO2 &in);

}  // namespace robot::liegroups::proto
