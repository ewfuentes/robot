
#include "common/liegroups/so2_to_proto.hh"

namespace robot::liegroups::proto {
void pack_into(const liegroups::SO2 &in, SO2 *out) {
    out->mutable_complex()->set_real(in.params()(0));
    out->mutable_complex()->set_imag(in.params()(1));
}

liegroups::SO2 unpack_from(const SO2 &in) {
    liegroups::SO2 out;
    out.data()[0] = in.complex().real();
    out.data()[1] = in.complex().imag();
    return out;
}

}  // namespace robot::liegroups::proto
