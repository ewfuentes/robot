
#include "common/math/multivariate_normal_cdf.hh"

#include <cmath>

#include "approxcdf/approxcdf.h"
#include "common/check.hh"

namespace robot::math {
std::optional<double> multivariate_normal_cdf(const Eigen::VectorXd &mu,
                                              const Eigen::MatrixXd &sigma,
                                              const Eigen::VectorXd &upper_bound,
                                              const bool return_log_p) {
    CHECK(sigma.rows() == sigma.cols());
    CHECK(sigma.rows() == mu.rows());
    CHECK(sigma.rows() == upper_bound.rows());

    const int dim = mu.rows();
    constexpr bool NONSTANDARD = false;
    const double result = norm_cdf(upper_bound.data(), sigma.data(), dim, mu.data(), dim,
                                   NONSTANDARD, return_log_p, nullptr);

    if (std::isnan(result)) {
        return std::nullopt;
    }
    return result;
}
}  // namespace robot::math
