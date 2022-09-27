
#include "experimental/beacon_sim/ekf_slam_estimate_to_proto.hh"

#include <algorithm>

#include "common/math/matrix_to_proto.hh"

namespace robot::experimental::beacon_sim::proto {
void pack_into(const beacon_sim::EkfSlamEstimate &in, EkfSlamEstimate *out) {
    pack_into(in.mean, out->mutable_mean());
    pack_into(in.cov, out->mutable_cov());

    out->clear_beacon_ids();
    for (const int id : in.beacon_ids) {
        out->add_beacon_ids(id);
    }
}

beacon_sim::EkfSlamEstimate unpack_from(const EkfSlamEstimate &in) {
    beacon_sim::EkfSlamEstimate out;
    out.mean = unpack_from<Eigen::VectorXd>(in.mean());
    out.cov = unpack_from<Eigen::MatrixXd>(in.cov());

    std::copy(in.beacon_ids().begin(), in.beacon_ids().end(), std::back_inserter(out.beacon_ids));

    return out;
}
}  // namespace robot::experimental::beacon_sim::proto
