
#include "experimental/beacon_sim/obstacle.hh"

#include "common/liegroups/se2.hh"

namespace robot::experimental::beacon_sim {

Obstacle::Obstacle(std::vector<Eigen::Vector2d> pts_in_frame)
    : pts_in_frame_{std::move(pts_in_frame)} {}

bool Obstacle::is_inside(const Eigen::Vector2d &query_in_frame) const {
    // A point in polygon check is run by counting the number of intesections between an
    // arbitrary ray originating at the query point and the edges of the polygon. If the number of
    // intersections is even, then the query point is outside of the polygon.
    // Define the arbitrary ray to be along the +X axis in the frame. Then an intersection occurs if
    // one of the segment endpoints is above the +X axis and the other is below.
    int intersection_counts = 0;
    const liegroups::SE2 query_from_frame = liegroups::SE2(0.0, query_in_frame).inverse();
    apply([&intersection_counts, &query_from_frame](const Eigen::Vector2d &a_in_frame,
                                                    const Eigen::Vector2d &b_in_frame) mutable {
        const Eigen::Vector2d a_in_query = query_from_frame * a_in_frame;
        const Eigen::Vector2d b_in_query = query_from_frame * b_in_frame;

        // If both endpoints are above or below the ray, then the product of the lateral distance
        // from the ray will be positive. If the product is negative, it implies that the segment
        // intersects the line.
        const bool straddles_x_axis = a_in_query.y() * b_in_query.y() < 0.0;
        if (straddles_x_axis) {
            // Check if the intersection occurs before or after the ray
            const double intersection_frac =
                std::abs(a_in_query.y()) / std::abs((b_in_query - a_in_query).y());
            const Eigen::Vector2d intersection_in_query =
                a_in_query + intersection_frac * (b_in_query - a_in_query);
            if (intersection_in_query.x() > 0.0) {
                intersection_counts++;
            }
        }

        return true;
    });

    return (intersection_counts % 2) == 1;
};

}  // namespace robot::experimental::beacon_sim
