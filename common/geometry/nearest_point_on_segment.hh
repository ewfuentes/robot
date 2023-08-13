
#include "Eigen/Core"

namespace robot::geometry {

template <int DIM>
struct NearestPointOnSegmentResult {
    double frac;
    Eigen::Vector<double, DIM> nearest_pt_in_frame;
};

template <int DIM>
NearestPointOnSegmentResult<DIM> nearest_point_on_segment_result(
    const Eigen::Vector<double, DIM> &start_in_frame,
    const Eigen::Vector<double, DIM> &end_in_frame,
    const Eigen::Vector<double, DIM> &query_in_frame) {
    using Vec = Eigen::Vector<double, DIM>;

    const Vec end_in_start = end_in_frame - start_in_frame;
    const double end_from_start_dist = end_in_start.norm();
    const Vec query_in_start = query_in_frame - start_in_frame;
    const double frac =
        query_in_start.dot(end_in_start) / (end_from_start_dist * end_from_start_dist);
    if (frac > 1) {
        return {
            .frac = 1.0,
            .nearest_pt_in_frame = end_in_frame,
        };
    } else if (frac < 0) {
        return {
            .frac = 0.0,
            .nearest_pt_in_frame = start_in_frame,
        };
    }
    const Vec nearest_point_in_frame = (1 - frac) * start_in_frame + frac * end_in_frame;
    return {
        .frac = frac,
        .nearest_pt_in_frame = nearest_point_in_frame,
    };
}
}  // namespace robot::geometry
