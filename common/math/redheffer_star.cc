
#include "common/math/redheffer_star.hh"

#include "Eigen/Dense"

namespace robot::math {

Eigen::MatrixXd scattering_from_transfer(const Eigen::MatrixXd &a) {
    // Break down the matrix into blockwise elements
    // a = [[A B]
    //      [C D]]
    const int dim = a.rows() / 2;
    const Eigen::MatrixXd A_11 = a.topLeftCorner(dim, dim);
    const Eigen::MatrixXd A_12 = a.topRightCorner(dim, dim);
    const Eigen::MatrixXd A_21 = a.bottomLeftCorner(dim, dim);
    const Eigen::MatrixXd A_22 = a.bottomRightCorner(dim, dim);

    const Eigen::MatrixXd A_11_inv = A_11.inverse();

    return (Eigen::MatrixXd(2 * dim, 2 * dim) << A_11_inv, -A_11_inv * A_12, A_21 * A_11_inv,
            A_22 - A_21 * A_11_inv * A_12)
        .finished();
}

Eigen::MatrixXd transfer_from_scattering(const Eigen::MatrixXd &a) {
    // The same transformation brings us back
    return scattering_from_transfer(a);
}

Eigen::MatrixXd redheffer_star(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
    // Break down the matrices into blockwise elements
    // a = [[A B]
    //      [C D]]
    // b = [[W X]
    //      [Y Z]]
    const int dim = a.rows() / 2;
    const Eigen::MatrixXd A_11 = a.topLeftCorner(dim, dim);
    const Eigen::MatrixXd A_12 = a.topRightCorner(dim, dim);
    const Eigen::MatrixXd A_21 = a.bottomLeftCorner(dim, dim);
    const Eigen::MatrixXd A_22 = a.bottomRightCorner(dim, dim);

    const Eigen::MatrixXd B_11 = b.topLeftCorner(dim, dim);
    const Eigen::MatrixXd B_12 = b.topRightCorner(dim, dim);
    const Eigen::MatrixXd B_21 = b.bottomLeftCorner(dim, dim);
    const Eigen::MatrixXd B_22 = b.bottomRightCorner(dim, dim);

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim, dim);
    const Eigen::MatrixXd I_min_A_12_B_21_inv = (I - A_12 * B_21).inverse();
    const Eigen::MatrixXd I_min_B_21_A_12_inv = (I - B_21 * A_12).inverse();

    return (Eigen::MatrixXd(2 * dim, 2 * dim) << B_11 * I_min_A_12_B_21_inv * A_11,
            B_12 + B_11 * I_min_A_12_B_21_inv * A_12 * B_22,
            A_21 + A_22 * I_min_B_21_A_12_inv * B_21 * A_11, A_22 * I_min_B_21_A_12_inv * B_22)
        .finished();
}
}  // namespace robot::math
