
#include "common/liegroups/se2.hh"
#include "common/liegroups/se2.pb.h"

namespace robot::liegroups::proto {

void pack_into(const liegroups::SE2 &in, SE2 *out);
liegroups::SE2 unpack_from(const SE2 &in);

}  // namespace robot::liegroups::proto
