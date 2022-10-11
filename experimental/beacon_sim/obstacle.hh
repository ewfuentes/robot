
#pragma once

#include <vector>

#include "Eigen/Core"

namespace robot::experimental::beacon_sim {
class Obstacle {
    // This class represents an obstacle, where an obstacle is a closed polygon. In order to ensure
    // that "inside" and "outside" are well defined, there is an implicit side between the first
    // and the last point.

    // Currently only supports convex polygons

   public:
    explicit Obstacle(std::vector<Eigen::Vector2d> pts_in_frame);

    bool is_inside(const Eigen::Vector2d &query_in_frame) const;

    // Apply a function to each edge of the obstacle. The function must have a signature of
    // bool(const Eigen::Vector2d &a, const Eigen::Vector2d &b). The iteration will stop if
    // the return value is false
    template <typename Callable>
    void apply(Callable f) const {
        const int num_pts = pts_in_frame_.size();
        for (int idx = 0; idx < num_pts; idx++) {
            const int next_idx = idx < num_pts - 1 ? idx + 1 : 0;
            if (!f(pts_in_frame_.at(idx), pts_in_frame_.at(next_idx))) {
                break;
            }
        }
    }

    const std::vector<Eigen::Vector2d> &pts_in_frame() const { return pts_in_frame_; }

   private:
    std::vector<Eigen::Vector2d> pts_in_frame_;
};
}  // namespace robot::experimental::beacon_sim
