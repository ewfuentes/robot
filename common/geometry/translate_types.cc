#include "common/geometry/translate_types.hh"

namespace robot::geometry {

cv::Mat eigen_mat_to_cv(const Eigen::MatrixXd &matrix) {
    cv::Mat result(matrix.rows(), matrix.cols(), CV_64F);  // Ensure correct type (double precision)
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            result.at<double>(i, j) = matrix(i, j);
        }
    }
    return result;
}

cv::Mat eigen_vec_to_cv(const Eigen::VectorXd &vector) {
    // cols should always be 1
    cv::Mat result(vector.rows(), vector.cols(), CV_64F);
    for (int i = 0; i < vector.rows(); i++) {
        result.at<double>(i, 0) = vector(i);
    }
    return result;
}

Eigen::MatrixXd cv_to_eigen_mat(const cv::Mat &matrix) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(matrix.rows, matrix.cols);
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            result(i, j) = matrix.at<double>(i, j);
        }
    }
    return result;
}

}  // namespace robot::geometry