
#include "planning/belief_road_map.hh"

namespace robot::planning::detail {
std::vector<int> find_nearest_node_idxs(const RoadMap &road_map, const Eigen::Vector2d &state,
                                        const int num_to_find) {
    std::vector<int> idxs(road_map.points.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(),
              [&state, &points = road_map.points](const int &a_idx, const int &b_idx) {
                  return (points.at(a_idx) - state).squaredNorm() <
                         (points.at(b_idx) - state).squaredNorm();
              });

    return std::vector<int>(idxs.begin(),
                            idxs.begin() + std::min(num_to_find, static_cast<int>(idxs.size())));
}
}  // namespace robot::planning::detail
