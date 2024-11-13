#include <iostream>
#include <vector>

#include "Eigen/Core"
#include "common/geometry/camera_geometry.hh"
#include "gtest/gtest.h"
#include "gtsam/geometry/Point3.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/Values.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"
#include "gtsam/slam/ProjectionFactor.h"

class GtsamTestHelper {
   public:
    static bool pixel_in_range(Eigen::Vector2d pixel, size_t img_width, size_t img_height) {
        return pixel[0] > 0 && pixel[0] < img_width && pixel[1] > 0 && pixel[1] < img_height;
    }
};

namespace robot::experimental::gtsam_testing {

TEST(GtsamTesting, gtsam_simple_cube) {
    std::vector<gtsam::Point3> cube_W;
    float cube_size = 1.0f;
    cube_W.push_back(gtsam::Point3(0, 0, 0));
    cube_W.push_back(gtsam::Point3(cube_size, 0, 0));
    cube_W.push_back(gtsam::Point3(cube_size, cube_size, 0));
    cube_W.push_back(gtsam::Point3(0, cube_size, 0));
    cube_W.push_back(gtsam::Point3(0, 0, cube_size));
    cube_W.push_back(gtsam::Point3(cube_size, 0, cube_size));
    cube_W.push_back(gtsam::Point3(cube_size, cube_size, cube_size));
    cube_W.push_back(gtsam::Point3(0, cube_size, cube_size));

    const size_t img_width = 640;
    const size_t img_height = 480;

    const double fx = 500.0;
    const double fy = fx;
    const double cx = img_width / 2.0;
    const double cy = img_height / 2.0;

    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;

    gtsam::Cal3_S2::shared_ptr K(new gtsam::Cal3_S2(fx, fy, 0, cx, cy));
    initial.insert(gtsam::Symbol('K', 0), *K);

    auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.1));
    auto measurementNoise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);  // For 2D measurements

    std::vector<gtsam::Pose3> poses;

    Eigen::Matrix3d rotation0(
        Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d(1, 0, 0)).toRotationMatrix() *
        Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d(0, -1, 0)).toRotationMatrix());
    gtsam::Pose3 pose0(gtsam::Rot3(rotation0), gtsam::Point3(4, 0, 0));
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(gtsam::Symbol('x', 0), pose0, poseNoise);
    poses.push_back(pose0);

    gtsam::Rot3 rotation1(Eigen::AngleAxis(M_PI / 6, Eigen::Vector3d(0, 0, 1)).toRotationMatrix());
    gtsam::Pose3 pose1(rotation1 * pose0.rotation(), rotation1 * pose0.translation());
    poses.push_back(pose1);

    gtsam::Pose3 pose2(
        gtsam::Rot3(rotation0 *
                    Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix()),
        gtsam::Point3(0, 4, 0));
    poses.push_back(pose2);

    for (size_t i = 0; i < poses.size(); i++) {
        initial.insert(gtsam::Symbol('x', i), poses[i]);
        if (i < poses.size() - 1)
            graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                gtsam::Symbol('x', i), gtsam::Symbol('x', i + 1), poses[i].between(poses[i + 1]),
                poseNoise);
    }
    for (size_t i = 0; i < cube_W.size(); i++) {
        initial.insert(gtsam::Symbol('L', i), cube_W[i]);
        for (size_t j = 0; j < poses.size(); j++) {
            gtsam::Point3 p_cube_C(poses[i].transformFrom(cube_W[j]));
            gtsam::Point2 pixel_p_cube_C(
                robot::geometry::project_point_to_camera_pixel(p_cube_C, fx, fy, cx, cy));
            if (GtsamTestHelper::pixel_in_range(pixel_p_cube_C, img_width, img_height)) {
                graph.emplace_shared<
                    gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(
                    pixel_p_cube_C, measurementNoise, gtsam::Symbol('x', j), gtsam::Symbol('L', i),
                    K);
            }
        }
    }

    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial);
    gtsam::Values result = optimizer.optimize();

    // result.print("Optimized Results:\n");

    constexpr double TOL = 1e-6;
    for (size_t i = 0; i < poses.size(); i++) {
        gtsam::Pose3 result_pose = result.at<gtsam::Pose3>(gtsam::Symbol('x', i));
        EXPECT_NEAR(poses[i].translation()[0], result_pose.translation()[0], TOL);
        EXPECT_NEAR(poses[i].translation()[1], result_pose.translation()[1], TOL);
        EXPECT_NEAR(poses[i].translation()[2], result_pose.translation()[2], TOL);
        EXPECT_TRUE(poses[i].rotation().matrix().isApprox(result_pose.rotation().matrix(), TOL));
    }
    for (size_t i = 0; i < cube_W.size(); i++) {
        gtsam::Point3 result_point = result.at<gtsam::Point3>(gtsam::Symbol('L', i));
        EXPECT_NEAR(cube_W[i][0], result_point[0], TOL);
        EXPECT_NEAR(cube_W[i][1], result_point[1], TOL);
        EXPECT_NEAR(cube_W[i][2], result_point[2], TOL);
    }
}
}  // namespace robot::experimental::gtsam_testing