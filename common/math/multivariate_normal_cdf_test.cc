
#include "common/math/multivariate_normal_cdf.hh"

#include "gtest/gtest.h"

namespace robot::math {
TEST(MultivariateNormalCdfTest, basic_test) {
    // Setup
    const Eigen::Vector3d mu = {1, 2, 3};
    const Eigen::Matrix3d sigma = Eigen::Matrix3d::Identity();

    // Action
    constexpr bool COMPUTE_NORMAL_PROB = false;
    const std::optional<double> result =
        multivariate_normal_cdf(mu, sigma, mu, COMPUTE_NORMAL_PROB);

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_TRUE(result.has_value());
    EXPECT_NEAR(result.value(), 0.125, TOL);
}
}  // namespace robot::math
