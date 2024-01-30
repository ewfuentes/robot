
#pragma once

#include <optional>

#include "Eigen/Core"

namespace robot::math {

// Compute the probability P(x_0 < ub[0], x_1 < ub[1], ..., x_n < ub[n))
// where the underlying probability density function is a multivariate normal
// with mean mu and covariance sigma.
//
// Returns std::nullopt if an error occurs.
std::optional<double> multivariate_normal_cdf(const Eigen::VectorXd &mu,
                                              const Eigen::MatrixXd &sigma,
                                              const Eigen::VectorXd &upper_bound,
                                              const bool return_log_p = false);
}  // namespace robot::math
