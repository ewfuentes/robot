#include <vector>

#include "Eigen/Dense"
#include "experimental/learn_descriptors/spatial_test_scene_cube.hh"
#include "gtest/gtest.h"
#include "visualization/opencv/opencv_viz.hh"

namespace robot::experimental::learn_descriptors {
TEST(SpatialTestSceneCubeTest, viz_cube) {
    SpatialTestSceneCube test_scene(1.0f);
    // geometry::viz_scene(std::vector<Eigen::Isometry3d>(), test_scene.get_points());
}

TEST(SpatialTestSceneCubeTest, viz_cube_with_cameras) {
    SpatialTestSceneCube test_scene(1.0f);

    const size_t img_width = 640;
    const size_t img_height = 480;
    const double fx = 500.0;
    const double fy = fx;
    const double cx = img_width / 2.0;
    const double cy = img_height / 2.0;
    gtsam::Cal3_S2::shared_ptr K(new gtsam::Cal3_S2(fx, fy, 0, cx, cy));

    test_scene.add_rand_cameras_face_origin(7, 2.0, 6.0, *K);

    std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras = test_scene.get_cameras();
    std::vector<Eigen::Isometry3d> camera_poses;
    camera_poses.reserve(cameras.size());
    for (const gtsam::PinholeCamera<gtsam::Cal3_S2> &T_world_cam : cameras) {
        camera_poses.emplace_back(T_world_cam.pose().matrix());
    }

    geometry::viz_scene(camera_poses, test_scene.get_points());
}
}  // namespace robot::experimental::learn_descriptors