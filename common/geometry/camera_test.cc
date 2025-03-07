#include "common/geometry/camera.hh"

#include "common/geometry/opencv_viz.hh"
#include "common/geometry/translate_types.hh"
#include "gtest/gtest.h"

class CameraTestHelper {
   public:
    static bool pixel_in_range(Eigen::Vector2d pixel, size_t img_width, size_t img_height) {
        return pixel[0] > 0 && pixel[0] < img_width && pixel[1] > 0 && pixel[1] < img_height;
    }
};

namespace robot::geometry {
TEST(CameraTest, test_get_intrinsics) {
    const double fx = 1., fy = 1., s = 0, px = 0.5, py = 0.5;
    const gtsam::Cal3_S2 K(fx, fy, s, px, py);
    const Eigen::Matrix3d intrinsic_mat = get_intrinsic_matrix(K);

    EXPECT_FLOAT_EQ(fx, intrinsic_mat(0, 0));
    EXPECT_FLOAT_EQ(fy, intrinsic_mat(1, 1));
    EXPECT_FLOAT_EQ(s, intrinsic_mat(0, 1));
    EXPECT_FLOAT_EQ(px, intrinsic_mat(0, 2));
    EXPECT_FLOAT_EQ(py, intrinsic_mat(1, 2));
}

TEST(CameraTest, test_projection) {
    const double fx = 1., fy = 1., s = 0, px = 0.5, py = 0.5;
    const gtsam::Cal3_S2 K(fx, fy, s, px, py);
    const Eigen::Matrix3d intrinsic_mat = get_intrinsic_matrix(K);

    const Eigen::Vector3d p_cam_lmk1(0.1, 0.15, 1.2);
    const Eigen::Vector3d pxl_homog_cam_lmk1 = project(intrinsic_mat, p_cam_lmk1);
    const Eigen::Vector2d pxl_inhomog_cam_lmk1 =
        pxl_homog_cam_lmk1.head<2>() / pxl_homog_cam_lmk1(2);
    const Eigen::Vector3d p_cam_lmk1_deproject = deproject(intrinsic_mat, pxl_homog_cam_lmk1);
    const Eigen::Vector3d p_cam_lmk1_deproject_inhomog =
        deproject(intrinsic_mat, pxl_inhomog_cam_lmk1, p_cam_lmk1(2));

    EXPECT_TRUE(p_cam_lmk1.isApprox(p_cam_lmk1_deproject));
    EXPECT_TRUE(p_cam_lmk1.isApprox(p_cam_lmk1_deproject_inhomog));
}

TEST(CameraTest, test_estimate_pose) {
    std::vector<Eigen::Vector3d> p_W_cube;
    float cube_size = 1.0f;
    p_W_cube.push_back(Eigen::Vector3d(0, 0, 0));
    p_W_cube.push_back(Eigen::Vector3d(cube_size, 0, 0));
    p_W_cube.push_back(Eigen::Vector3d(cube_size, cube_size, 0));
    p_W_cube.push_back(Eigen::Vector3d(0, cube_size, 0));
    p_W_cube.push_back(Eigen::Vector3d(0, 0, cube_size));
    p_W_cube.push_back(Eigen::Vector3d(cube_size, 0, cube_size));
    p_W_cube.push_back(Eigen::Vector3d(cube_size, cube_size, cube_size));
    p_W_cube.push_back(Eigen::Vector3d(0, cube_size, cube_size));

    Eigen::Matrix3d R_new_points(
        Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d(0, 0, 1)).toRotationMatrix() *
        Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d(1, 0, 0)).toRotationMatrix());

    const int initial_size = p_W_cube.size();
    const Eigen::Vector3d p_world_cube_center(cube_size / 2, cube_size / 2, cube_size / 2);
    for (const Eigen::Vector3d &point_W_cube : p_W_cube) {
        p_W_cube.emplace_back(R_new_points * (point_W_cube - p_world_cube_center) +
                              p_world_cube_center);
    }

    const size_t img_width = 640;
    const size_t img_height = 480;
    const double fx = 500.0;
    const double fy = fx;
    const double cx = img_width / 2.0;
    const double cy = img_height / 2.0;

    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    Eigen::Matrix3d K_eig = cv_to_eigen_mat(K);
    std::vector<Eigen::Isometry3d> poses;

    Eigen::Matrix3d R_world_cam0(
        Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix() *
        Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)).toRotationMatrix());
    Eigen::Isometry3d T_world_cam0 = Eigen::Isometry3d::Identity();
    T_world_cam0.linear() = R_world_cam0;
    T_world_cam0.translation() = Eigen::Vector3d(4, cube_size / 2, cube_size / 2);
    poses.push_back(T_world_cam0);

    Eigen::Matrix3d R_world_cam0_to_cam1(
        Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d(0, 0, 1)).toRotationMatrix());
    Eigen::Isometry3d T_world_cam1;
    T_world_cam1.linear() = R_world_cam0_to_cam1 * T_world_cam0.linear();
    T_world_cam1.translation() =
        R_world_cam0_to_cam1 * (T_world_cam0.translation() - p_world_cube_center) +
        p_world_cube_center;
    poses.push_back(T_world_cam1);

    std::vector<cv::KeyPoint> kpts0;
    std::vector<cv::KeyPoint> kpts1;
    std::vector<cv::DMatch> matches;

    for (const Eigen::Vector3d &point_W_cube : p_W_cube) {
        Eigen::Vector3d p_cam0_cubep =
            (T_world_cam0.inverse() *
             Eigen::Vector4d(point_W_cube(0), point_W_cube(1), point_W_cube(2), 1.))
                .head<3>();
        Eigen::Vector3d p_cam1_cubep =
            (T_world_cam1.inverse() *
             Eigen::Vector4d(point_W_cube(0), point_W_cube(1), point_W_cube(2), 1.))
                .head<3>();
        Eigen::Vector3d pxl_c0_pcube_homog = K_eig * p_cam0_cubep;
        Eigen::Vector3d pxl_c1_pcube_homog = K_eig * p_cam1_cubep;
        Eigen::Vector2d pxl_c0_pcube = pxl_c0_pcube_homog.head<2>() / pxl_c0_pcube_homog(2);
        Eigen::Vector2d pxl_c1_pcube = pxl_c1_pcube_homog.head<2>() / pxl_c1_pcube_homog(2);
        if (CameraTestHelper::pixel_in_range(pxl_c0_pcube, img_width, img_height) &&
            CameraTestHelper::pixel_in_range(pxl_c1_pcube, img_width, img_height)) {
            kpts0.push_back(cv::KeyPoint(pxl_c0_pcube[0], pxl_c0_pcube[1], 3));
            kpts1.push_back(cv::KeyPoint(pxl_c1_pcube[0], pxl_c1_pcube[1], 3));
            matches.emplace_back(cv::DMatch(kpts0.size() - 1, kpts1.size() - 1, 0));
        }
    }

    Eigen::Isometry3d T_cam0_cam1_estimate = estimate_c0_c1(kpts0, kpts1, matches, K);
    Eigen::Isometry3d T_cam0_cam1_scale = T_world_cam0.inverse() * T_world_cam1;
    T_cam0_cam1_scale.translation() /= T_cam0_cam1_scale.translation().norm();
    EXPECT_TRUE(
        T_cam0_cam1_estimate.translation().isApprox(T_cam0_cam1_scale.translation(), 0.000001));
    EXPECT_TRUE(T_cam0_cam1_estimate.linear().isApprox(T_cam0_cam1_scale.linear(), 0.001));

    poses.emplace_back(T_world_cam0 * T_cam0_cam1_estimate);
    // viz_scene(poses, p_W_cube);
}
}  // namespace robot::geometry