
#pragma once

#include "common/liegroups/so3.hh"
#include "common/liegroups/so3.pb.h"

namespace robot::liegroups::proto {

void pack_into(const liegroups::SO3 &in, SO3 *out);
liegroups::SO3 unpack_from(const SO3 &in);

}  // namespace robot::liegroups::proto
