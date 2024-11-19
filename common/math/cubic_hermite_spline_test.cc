
#include "common/math/cubic_hermite_spline.hh"

#include "gtest/gtest.h"

namespace robot::math {

TEST(CubicHermiteSplineTest, inputs_must_be_sorted) {
    // Setup
    const std::vector<double> ts{0.0, 1.0, 3.0, 2.0, 4.0};
    const std::vector<double> xs{10.0, 20.0, 40.0, 30.0, 50.0};

    // Action + Verification
    EXPECT_THROW(CubicHermiteSpline(ts, xs), check_failure);
}

TEST(CubicHermiteSplineTest, query_after_bounds_throws) {
    // Setup
    const std::vector<double> ts{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> xs{10.0, 20.0, 40.0, 30.0, 50.0};
    const auto spline = CubicHermiteSpline(ts, xs);

    // Action + Verification
    EXPECT_THROW(spline(5.0), check_failure);
}

TEST(CubicHermiteSplineTest, query_before_bounds_throws) {
    // Setup
    const std::vector<double> ts{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<double> xs{10.0, 20.0, 40.0, 30.0, 50.0};
    const auto spline = CubicHermiteSpline(ts, xs);

    // Action + Verification
    EXPECT_THROW(spline(-5.0), check_failure);
}

TEST(CubicHermiteSplineTest, linear_points_produce_linear_result) {
    // Setup
    const std::vector<double> ts{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> xs{10.0, 20.0, 30.0, 40.0};
    const auto spline = CubicHermiteSpline(ts, xs);

    // Action + Verification
    constexpr double TOL = 1e-6;
    EXPECT_NEAR(spline(0.5), 15.0, TOL);
    EXPECT_NEAR(spline(1.5), 25.0, TOL);
    EXPECT_NEAR(spline(2.5), 35.0, TOL);
}

TEST(CubicHermiteSplineTest, nonuniform_linear_points_produce_linear_result) {
    // Setup
    const std::vector<double> ts{0.0, 1.0, 2.0, 4.0};
    const std::vector<double> xs{10.0, 20.0, 30.0, 50.0};
    const auto spline = CubicHermiteSpline(ts, xs);

    // Action + Verification
    constexpr double TOL = 1e-6;
    EXPECT_NEAR(spline(0.5), 15.0, TOL);
    EXPECT_NEAR(spline(1.5), 25.0, TOL);
    EXPECT_NEAR(spline(2.5), 35.0, TOL);
}

TEST(CubicHermiteSplineTest, quadratic_points_produce_quadratic_result) {
    // Setup
    const std::vector<double> ts{-1.0, 0.0, 1.0, 2.0};
    const std::vector<double> xs{2.0, 1.0, 2.0, 5.0};
    const auto spline = CubicHermiteSpline(ts, xs);

    // Action + Verification
    constexpr double TOL = 1e-6;
    EXPECT_NEAR(spline(0.0), 1.0, TOL);
    EXPECT_NEAR(spline(0.25), 1.0625, TOL);
    EXPECT_NEAR(spline(0.5), 1.25, TOL);
    EXPECT_NEAR(spline(1.0), 2.0, TOL);
}
}  // namespace robot::math
