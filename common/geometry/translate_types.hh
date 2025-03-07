#pragma once
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "opencv2/opencv.hpp"

namespace robot::geometry {
cv::Mat eigen_mat_to_cv(const Eigen::MatrixXd &matrix);
cv::Mat eigen_vec_to_cv(const Eigen::VectorXd &vector);
Eigen::MatrixXd cv_to_eigen_mat(const cv::Mat &matrix);
}  // namespace robot::geometry