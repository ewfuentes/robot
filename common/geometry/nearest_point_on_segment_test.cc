
#include "common/geometry/nearest_point_on_segment.hh"

#include "gtest/gtest.h"

namespace robot::geometry {
TEST(NearestPointOnSegmentTest, point_on_segment) {
    // Setup
    const Eigen::Vector2d START_IN_FRAME{3, 4};
    const Eigen::Vector2d END_IN_FRAME{10, 20};
    const double EXPECTED_FRAC = 0.37;
    const Eigen::Vector2d QUERY_IN_FRAME =
        (1 - EXPECTED_FRAC) * START_IN_FRAME + EXPECTED_FRAC * END_IN_FRAME;

    // Action
    const auto result =
        nearest_point_on_segment_result(START_IN_FRAME, END_IN_FRAME, QUERY_IN_FRAME);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_NEAR(result.frac, EXPECTED_FRAC, TOL);
    EXPECT_NEAR(result.nearest_pt_in_frame.x(), QUERY_IN_FRAME.x(), TOL);
    EXPECT_NEAR(result.nearest_pt_in_frame.y(), QUERY_IN_FRAME.y(), TOL);
}

TEST(NearestPointOnSegmentTest, query_behind_start) {
    // Setup
    const Eigen::Vector2d START_IN_FRAME{3, 4};
    const Eigen::Vector2d END_IN_FRAME{10, 20};
    const double EXPECTED_FRAC = 0.0;
    const Eigen::Vector2d QUERY_IN_FRAME{0, 0};

    // Action
    const auto result =
        nearest_point_on_segment_result(START_IN_FRAME, END_IN_FRAME, QUERY_IN_FRAME);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_NEAR(result.frac, EXPECTED_FRAC, TOL);
    EXPECT_NEAR(result.nearest_pt_in_frame.x(), START_IN_FRAME.x(), TOL);
    EXPECT_NEAR(result.nearest_pt_in_frame.y(), START_IN_FRAME.y(), TOL);
}

TEST(NearestPointOnSegmentTest, query_beyond_end) {
    // Setup
    const Eigen::Vector2d START_IN_FRAME{3, 4};
    const Eigen::Vector2d END_IN_FRAME{10, 20};
    const double EXPECTED_FRAC = 1.0;
    const Eigen::Vector2d QUERY_IN_FRAME{100, 100};

    // Action
    const auto result =
        nearest_point_on_segment_result(START_IN_FRAME, END_IN_FRAME, QUERY_IN_FRAME);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_NEAR(result.frac, EXPECTED_FRAC, TOL);
    EXPECT_NEAR(result.nearest_pt_in_frame.x(), END_IN_FRAME.x(), TOL);
    EXPECT_NEAR(result.nearest_pt_in_frame.y(), END_IN_FRAME.y(), TOL);
}

}  // namespace robot::geometry
