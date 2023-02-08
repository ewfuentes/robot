
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
    // [c] = B' * A' [a]
    // [b]           [d]
    // Where A' and B' are the scattering forms of the transfer matrices A and B.
    // We check to see that given [[a], [d]], the scattering form produces [[c], [b]]

    // Setup
    const Eigen::Matrix2d A{{1.0, 2.0}, {3.0, 4.0}};
    const Eigen::Matrix2d B{{4.0, 5.0}, {6.0, 7.0}};
    const Eigen::Vector2d input{0.1, 0.2};
    const Eigen::Vector2d output = B * A * input;

    // Action
    const Eigen::Matrix2d A_scatter = scattering_from_transfer(A);
    const Eigen::Matrix2d B_scatter = scattering_from_transfer(B);
    const Eigen::Matrix2d combined = redheffer_star(B_scatter, A_scatter);
    const Eigen::Vector2d scatter_input{input.x(), output.y()};
    const Eigen::Vector2d scatter_output = combined * scatter_input;

    std::cout << "input" << std::endl << input <<std::endl;
    std::cout << "output" << std::endl << output << std::endl;

    std::cout << "A Scatter" << std::endl << A_scatter << std::endl;
    std::cout << "B Scatter" << std::endl << B_scatter << std::endl;
    std::cout << "combined" << std::endl << combined << std::endl;
    std::cout << "scatter input" << std::endl << scatter_input << std::endl;
    std::cout << "scatter output" << std::endl << scatter_output << std::endl;

    // Verification
    EXPECT_EQ(input.y(), scatter_output.x());
    EXPECT_EQ(output.x(), scatter_output.y());
}
}  // namespace robot::math
