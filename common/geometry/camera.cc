#include "common/geometry/camera.hh"

#include "common/geometry/translate_types.hh"

namespace robot::geometry {
Eigen::Matrix3d get_intrinsic_matrix(const gtsam::Cal3_S2 &intrinsic) {
    Eigen::Matrix3d K;
    K << intrinsic.fx(), intrinsic.skew(), intrinsic.px(), 0, intrinsic.fy(), intrinsic.py(), 0, 0,
        1;
    return K;
}

Eigen::Vector3d project(const Eigen::Matrix3d &K, const Eigen::Vector3d &p_cam_point) {
    return K * p_cam_point;
    // return (K * p_cam_point).head<2>();
}

Eigen::Vector3d deproject(const Eigen::Matrix3d &K, const Eigen::Vector3d &pixel_homog) {
    return K.inverse() * pixel_homog;
}

Eigen::Vector3d deproject(const Eigen::Matrix3d &K, const Eigen::Vector2d &pixel_inhomog,
                          const double depth) {
    return depth * K.inverse() * Eigen::Vector3d(pixel_inhomog(0), pixel_inhomog(1), 1.);
}

Eigen::Isometry3d estimate_cam0_from_cam1(const std::vector<cv::KeyPoint> &kpts0,
                                          const std::vector<cv::KeyPoint> &kpts1,
                                          const std::vector<cv::DMatch> &matches,
                                          const cv::Mat &K) {
    assert(kpts1.size() != 0 && kpts0.size() != 0);
    Eigen::Isometry3d result;
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (const cv::DMatch &match : matches) {
        pts1.push_back(kpts0[match.queryIdx].pt);
        pts2.push_back(kpts1[match.trainIdx].pt);
    }
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0);
    cv::Mat R_c1_c0, t_c1_c0;
    cv::recoverPose(E, pts1, pts2, K, R_c1_c0, t_c1_c0);
    result.linear() = cv_to_eigen_mat(R_c1_c0);
    result.translation() = cv_to_eigen_mat(t_c1_c0);
    result = result.inverse();
    return result;
}
}  // namespace robot::geometry