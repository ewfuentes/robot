

#include "common/liegroups/se3_to_proto.hh"

#include "common/liegroups/so3_to_proto.hh"
#include "common/math/matrix_to_proto.hh"

namespace robot::liegroups::proto {
  void pack_into(const liegroups::SE3 &in, SE3 *out) {
    pack_into(in.so3(), out->mutable_rotation());
    pack_into(in.translation(), out->mutable_translation());
  }

  liegroups::SE3 unpack_from(const SE3 &in) {
    liegroups::SE3 out;
    out.so3() = unpack_from(in.rotation());
    out.translation() = unpack_from<liegroups::SE3::TranslationType>(in.translation());
    return out;
  }
}  // namespace robot::liegroups::proto
