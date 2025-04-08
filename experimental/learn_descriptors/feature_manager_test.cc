#include "experimental/learn_descriptors/feature_manager.hh"

#include "Eigen/Dense"
#include "experimental/learn_descriptors/spatial_scene_test_cube.hh"
#include "gtest/gtest.h"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/PinholeCamera.h"
#include "gtsam/geometry/Point3.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Rot3.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
TEST(feature_manager_test, test) {
    FeatureManager feature_manager;
    SpatialSceneTestCube test_cube(1.f);

    const size_t img_width = 640;
    const size_t img_height = 480;
    const double fx = 500.0;
    const double fy = fx;
    const double cx = img_width / 2.0;
    const double cy = img_height / 2.0;

    gtsam::Cal3_S2::shared_ptr K(new gtsam::Cal3_S2(fx, fy, 0, cx, cy));

    Eigen::Matrix3d rotation0(
        Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix() *
        Eigen::AngleAxis(-M_PI / 2, Eigen::Vector3d(1, 0, 0)).toRotationMatrix());
    gtsam::Pose3 pose0(gtsam::Rot3(rotation0), gtsam::Point3(4, 0, 0));
    gtsam::PinholeCamera<gtsam::Cal3_S2> camera0(pose0, *K);

    gtsam::Pose3 pose1(
        gtsam::Rot3(Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix() *
                    rotation0),
        gtsam::Point3(0, 4, 0));
    gtsam::PinholeCamera<gtsam::Cal3_S2> camera1(pose1, *K);

    gtsam::Pose3 pose2(
        gtsam::Rot3(Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix() *
                    rotation0),
        gtsam::Point3(0, 4, 0));
    gtsam::PinholeCamera<gtsam::Cal3_S2> camera2(pose1, *K);

    std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras{camera0, camera1};

    std::vector<Eigen::Vector2d> projected_pixels =
        test_cube.get_projected_pixels(camera0, cv::Size(img_width, img_height));
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
    for (const Eigen::Vector2d &pt : projected_pixels) {
        cv::circle(image, cv::Point2i(static_cast<int>(pt[0]), static_cast<int>(pt[1])), 4,
                   cv::Scalar(0, 0, 0));
    }
    cv::imshow("projected cubes", image);
    cv::waitKey(0);

    std::vector<std::pair<gtsam::PinholeCamera<gtsam::Cal3_S2>, cv::Size>> cams{
        std::pair<gtsam::PinholeCamera<gtsam::Cal3_S2>, cv::Size>(camera0,
                                                                  cv::Size(img_width, img_height)),
        std::pair<gtsam::PinholeCamera<gtsam::Cal3_S2>, cv::Size>(camera1,
                                                                  cv::Size(img_width, img_height))};
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences =
        test_cube.get_corresponding_pixels(cams);
}
}  // namespace robot::experimental::learn_descriptors