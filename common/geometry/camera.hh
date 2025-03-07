#pragma once
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "gtsam/geometry/Cal3_S2.h"
#include "opencv2/opencv.hpp"

namespace robot::geometry {
Eigen::Matrix3d get_intrinsic_matrix(const gtsam::Cal3_S2 &intrinsic);

/// @return Homogeneous pixel projection coordinates
Eigen::Vector3d project(const Eigen::Matrix3d &K, const Eigen::Vector3d &p_cam_point);
/**
 * @param pixel_homog homogeneous pixel cordinates 3x1
 * @return 3d coordinate in camera frame
 */
Eigen::Vector3d deproject(const Eigen::Matrix3d &K, const Eigen::Vector3d &pixel_homog);
/**
 * @param pixel_inhomog inhomogeneous pixel cordinates 2x1
 * @return 3d coordinate in camera frame
 */
Eigen::Vector3d deproject(const Eigen::Matrix3d &K, const Eigen::Vector2d &pixel_inhomog,
                          const double depth);

Eigen::Isometry3d estimate_c0_c1(const std::vector<cv::KeyPoint> &kpts0,
                                 const std::vector<cv::KeyPoint> &kpts1,
                                 const std::vector<cv::DMatch> &matches, const cv::Mat &K);
}  // namespace robot::geometry