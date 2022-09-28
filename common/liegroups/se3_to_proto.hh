
#include "common/liegroups/se3.hh"
#include "common/liegroups/se3.pb.h"

namespace robot::liegroups::proto {

  void pack_into(const liegroups::SE3 &in, SE3 *out);
  liegroups::SE3 unpack_from(const SE3 &in);

}  // namespace robot::liegroups::proto
