
#include "common/liegroups/so3_to_proto.hh"

namespace robot::liegroups::proto {
void pack_into(const liegroups::SO3 &in, SO3 *out) {
    // The data is returned as q.imag[0], q.imag[1], q.imag[2], q.real
    out->mutable_quaternion()->set_a(in.params()[3]);
    out->mutable_quaternion()->set_b(in.params()[0]);
    out->mutable_quaternion()->set_c(in.params()[1]);
    out->mutable_quaternion()->set_d(in.params()[2]);
}

liegroups::SO3 unpack_from(const SO3 &in) {
    liegroups::SO3 out;
    out.data()[0] = in.quaternion().b();
    out.data()[1] = in.quaternion().c();
    out.data()[2] = in.quaternion().d();
    out.data()[3] = in.quaternion().a();
    return out;
}
}  // namespace robot::liegroups::proto
