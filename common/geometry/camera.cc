#include "common/geometry/camera.hh"

#include <exception>
#include <optional>

#include "common/check.hh"
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
}

Eigen::Vector3d deproject(const Eigen::Matrix3d &K, const Eigen::Vector3d &pixel_homog) {
    return K.inverse() * pixel_homog;
}

Eigen::Vector3d deproject(const Eigen::Matrix3d &K, const Eigen::Vector2d &pixel_inhomog,
                          const double depth) {
    return depth * K.inverse() * Eigen::Vector3d(pixel_inhomog(0), pixel_inhomog(1), 1.);
}

std::optional<Eigen::Isometry3d> estimate_cam0_from_cam1(const std::vector<cv::KeyPoint> &kpts0,
                                                         const std::vector<cv::KeyPoint> &kpts1,
                                                         const std::vector<cv::DMatch> &matches,
                                                         const cv::Mat &K) {
    ROBOT_CHECK(matches.size() >= 5 && kpts1.size() >= 5 && kpts0.size() >= 5);
    Eigen::Isometry3d result;
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    for (const cv::DMatch &match : matches) {
        pts1.push_back(kpts0[match.queryIdx].pt);
        pts2.push_back(kpts1[match.trainIdx].pt);
    }
    ROBOT_CHECK(pts1.size() == pts2.size() && pts1.size() >= 5);
    try {
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0);
        cv::Mat R_c1_c0, t_c1_c0;
        ROBOT_CHECK(!E.empty(), "Essential matrix is empty.");
        // TOOD: handle multiple returned candidate E matrices better (they are stacked on top of
        // each other in E)
        if (E.rows > 3) {
            E = E.rowRange(0, 3);
        }
        cv::recoverPose(E, pts1, pts2, K, R_c1_c0, t_c1_c0);
        result.linear() = cv_to_eigen_mat(R_c1_c0);
        result.translation() = cv_to_eigen_mat(t_c1_c0);
        result = result.inverse();
        return result;
    } catch (const std::exception &e) {
        std::cerr << "Failed to estimate pose up to scale cam0_from_cam1.\n"
                  << e.what() << std::endl;
        return std::nullopt;
    }
}
}  // namespace robot::geometry