#include <string>
#include <vector>

#include "Eigen/Dense"
#include "common/check.hh"
#include "experimental/learn_descriptors/spatial_test_scene_cube.hh"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/PinholeCamera.h"
#include "visualization/opencv/opencv_viz.hh"

using namespace robot::experimental::learn_descriptors;

int main() {
    SpatialTestSceneCube test_scene(1.0f);

    const size_t img_width = 640;
    const size_t img_height = 480;
    const double fx = 500.0;
    const double fy = fx;
    const double cx = static_cast<double>(img_width) / 2.0;
    const double cy = static_cast<double>(img_height) / 2.0;
    gtsam::Cal3_S2::shared_ptr K(new gtsam::Cal3_S2(fx, fy, 0, cx, cy));

    SpatialTestScene::Noise pose_noise({0.05, 0.05, 0.05, 0.15, 0.15, 0.15});
    test_scene.add_rand_cameras_face_origin(7, 2.0, 6.0, pose_noise, *K);

    std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras = test_scene.cameras();
    std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras_grndtrth =
        test_scene.cameras_groundtruth();
    ROBOT_CHECK(cameras.size() == cameras_grndtrth.size());
    std::vector<robot::visualization::VizPose> viz_cam_poses;
    viz_cam_poses.reserve(cameras.size() + cameras_grndtrth.size());
    for (size_t i = 0; i < cameras.size(); i++) {
        viz_cam_poses.emplace_back(Eigen::Isometry3d(cameras[i].pose().matrix()),
                                   "cam_" + std::to_string(i));
        viz_cam_poses.emplace_back(Eigen::Isometry3d(cameras_grndtrth[i].pose().matrix()),
                                   "cam_grnd_" + std::to_string(i));
    }
    ROBOT_CHECK(test_scene.points().size() == test_scene.points_groundtruth().size());
    std::vector<robot::visualization::VizPoint> viz_points;
    for (size_t i = 0; i < test_scene.points().size(); i++) {
        viz_points.emplace_back(test_scene.points()[i], "pt_" + std::to_string(i));
        viz_points.emplace_back(test_scene.points_groundtruth()[i], "pt_grnd_" + std::to_string(i));
    }
    robot::visualization::viz_scene(viz_cam_poses, viz_points);
}