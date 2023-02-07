
#include "planning/road_map_to_proto.hh"

#include "common/math/matrix_to_proto.hh"

namespace robot::planning::proto {
void pack_into(const planning::RoadMap &in, RoadMap *out) {
    for (const auto &pt : in.points) {
        pack_into(pt, out->mutable_points()->Add());
    }

    pack_into(in.adj, out->mutable_adj());
}

planning::RoadMap unpack_from(const RoadMap &in) {
  planning::RoadMap out;
  for (const auto &pt: in.points()) {
    out.points.push_back(unpack_from<Eigen::Vector2d>(pt));
  }

  out.adj = unpack_from<Eigen::MatrixXd>(in.adj());
  return out;
}
}  // namespace robot::planning::proto
