
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
    const Eigen::MatrixXd A = a.topLeftCorner(dim, dim);
    const Eigen::MatrixXd B = a.topRightCorner(dim, dim);
    const Eigen::MatrixXd C = a.bottomLeftCorner(dim, dim);
    const Eigen::MatrixXd D = a.bottomRightCorner(dim, dim);

    const Eigen::MatrixXd W = b.topLeftCorner(dim, dim);
    const Eigen::MatrixXd X = b.topRightCorner(dim, dim);
    const Eigen::MatrixXd Y = b.bottomLeftCorner(dim, dim);
    const Eigen::MatrixXd Z = b.bottomRightCorner(dim, dim);

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim, dim);
    const Eigen::MatrixXd W_I_min_BY_inv = W * (I - B * Y).inverse();
    const Eigen::MatrixXd D_I_min_YB_inv = D * (I - Y * B).inverse();

    return (Eigen::MatrixXd(2 * dim, 2 * dim) << W_I_min_BY_inv * A, X + W_I_min_BY_inv * B * Z,
            C + D_I_min_YB_inv * Y * A, D_I_min_YB_inv * Z)
        .finished();
}
}  // namespace robot::math
