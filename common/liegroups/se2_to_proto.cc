

#include "common/liegroups/se2_to_proto.hh"

#include "common/liegroups/so2_to_proto.hh"
#include "common/math/matrix_to_proto.hh"

namespace robot::liegroups::proto {
void pack_into(const liegroups::SE2 &in, SE2 *out) {
    pack_into(in.so2(), out->mutable_rotation());
    pack_into(in.translation(), out->mutable_translation());
}

liegroups::SE2 unpack_from(const SE2 &in) {
    liegroups::SE2 out;
    out.so2() = unpack_from(in.rotation());
    out.translation() = unpack_from<liegroups::SE2::TranslationType>(in.translation());
    return out;
}
}  // namespace robot::liegroups::proto
