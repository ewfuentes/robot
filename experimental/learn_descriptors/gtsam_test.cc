#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"
#include "gtsam/slam/ProjectionFactor.h"
#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Point3.h"
#include "gtsam/nonlinear/Values.h"
#include "gtsam/inference/Symbol.h"
#include "Eigen/Core"

#include "common/geometry/camera_geometry.hh"

#include <iostream>
#include <vector>

#include "gtest/gtest.h"

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
    cube_W.push_back(gtsam::Point3(0,0,0));
    cube_W.push_back(gtsam::Point3(cube_size,0,0));
    cube_W.push_back(gtsam::Point3(cube_size,cube_size,0));
    cube_W.push_back(gtsam::Point3(0,cube_size,0));
    cube_W.push_back(gtsam::Point3(0,0,cube_size));
    cube_W.push_back(gtsam::Point3(cube_size,0,cube_size));
    cube_W.push_back(gtsam::Point3(cube_size,cube_size,cube_size));
    cube_W.push_back(gtsam::Point3(0,cube_size,cube_size));

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
    if (initial.exists(gtsam::Symbol('K', 0)))
        std::cout << "Inserted landmark with key K0" << std::endl;
    else
        std::cout << "Landmark insertion with key K0 failed" << std::endl;   

    auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(0.1));
    // auto landmarkNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);  // For 3D points
    auto measurementNoise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);  // For 2D measurements

    std::vector<gtsam::Pose3> poses;

    gtsam::Pose3 pose0(gtsam::Rot3(Eigen::AngleAxis(M_PI, Eigen::Vector3d(0,0,1)).toRotationMatrix()), gtsam::Point3(2, 0, 0));
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(gtsam::Symbol('x', 0), pose0, poseNoise);
    poses.push_back(pose0);

    gtsam::Pose3 pose1(gtsam::Rot3(Eigen::AngleAxis(-M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix()), gtsam::Point3(0,2,0));
    poses.push_back(pose1);

    for (size_t i = 0; i < poses.size(); i++) {
        initial.insert(gtsam::Symbol('x', i), poses[i]);
        if (initial.exists(gtsam::Symbol('x', i)))
            std::cout << "Inserted pose with key x" << i << std::endl;
        else
            std::cout << "Pose insertion with key x" << i << " failed" << std::endl;
        if (i < poses.size()-1)
            graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(gtsam::Symbol('x',i), gtsam::Symbol('x',i+1), poses[i].between(poses[i+1]), poseNoise);
    }
    // Add landmark observation factors
    for (size_t i = 0; i < cube_W.size(); i++) {
        initial.insert(gtsam::Symbol('L', i), cube_W[i]);
        if (initial.exists(gtsam::Symbol('L', i)))
            std::cout << "Inserted landmark with key L" << i << std::endl;
        else
            std::cout << "Landmark insertion with key L" << i << " failed" << std::endl;        
        // std::cout <<  "Inserted pose with key L" << i << std::endl;
        for (size_t j = 0; j < poses.size(); j++) {
            gtsam::Point3 p_cube_C(poses[i].transformFrom(cube_W[j]));
            gtsam::Point2 pixel_p_cube_C(robot::geometry::project_point_to_camera_pixel(p_cube_C, fx, fy, cx, cy));
            if (GtsamTestHelper::pixel_in_range(pixel_p_cube_C, img_width, img_height)) {
                graph.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(
                    pixel_p_cube_C, measurementNoise, gtsam::Symbol('x', j), gtsam::Symbol('L', i), K); 
            }
        }
    }

    // Run the optimizer to optimize poses, landmark positions, and calibration parameters
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial);
    gtsam::Values result = optimizer.optimize();

    // Print the optimized calibration parameters
    std::cout << "Optimized Calibration Parameters:\n";
    result.at<gtsam::Cal3_S2>(gtsam::Symbol('K', 0)).print("Calibration:");
    

    // Print the optimized landmark position and poses
    result.print("Optimized Results:\n");
}
}  // namespace robot::experimental::gtsam