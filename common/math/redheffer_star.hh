
#include "Eigen/Core"

namespace robot::math {

// Given a transfer matrix, create the equivalent scattering matrix
Eigen::MatrixXd scattering_from_transfer(const Eigen::MatrixXd &a);
Eigen::MatrixXd transfer_from_scattering(const Eigen::MatrixXd &a);

// Given two scattering matrices A and B, compute A * B
Eigen::MatrixXd redheffer_star(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);

}  // namespace robot::math
