
#include "common/math/redheffer_star.hh"

#include <iostream>

#include "gtest/gtest.h"

namespace robot::math {
TEST(RedhefferStarTest, simple_test) {
    // We should get the same result using transfer matrices as we do scattering matrices
    // The transfer form of the equations produce
    // [c] = B * A [a]
    // [d]         [b]
    // The scattering forms of the equations are:
    // [a] = B' * A' [c]
    // [d]           [b]
    // Where A' and B' are the scattering forms of the transfer matrices A and B.
    // We check to see that given [[a], [d]], the scattering form produces [[c], [b]]

    // Setup
    constexpr double TOL = 1e-6;
    const Eigen::Matrix2d A{{1.0, 2.0}, {3.0, 4.0}};
    const Eigen::Matrix2d B{{5.0, 6.0}, {7.0, 8.0}};
    const Eigen::Vector2d input{0.2, 0.3};
    const Eigen::Vector2d output = B * A * input;

    // Action
    const Eigen::Matrix2d A_scatter = scattering_from_transfer(A);
    const Eigen::Matrix2d B_scatter = scattering_from_transfer(B);
    const Eigen::Matrix2d combined = redheffer_star(B_scatter, A_scatter);
    const Eigen::Vector2d scatter_input{output.x(), input.y()};
    const Eigen::Vector2d scatter_output = combined * scatter_input;

    // Verification
    EXPECT_NEAR(input.x(), scatter_output.x(), TOL);
    EXPECT_NEAR(output.y(), scatter_output.y(), TOL);
}
}  // namespace robot::math
