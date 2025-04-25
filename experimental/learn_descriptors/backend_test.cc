#include "experimental/learn_descriptors/backend.hh"

#include "experimental/learn_descriptors/feature_manager.hh"
#include "gtest/gtest.h"
#include "experimental/learn_descriptors/spatial_test_scene_cube.hh"

namespace robot::experimental::learn_descriptors {
TEST(BackendTest, cube) {
    SpatialTestSceneCube test_scene(1.0f);

    const size_t img_width = 640;
    const size_t img_height = 480;
    const cv::Size img_size(640, 480);
    const double fx = 500.0;
    const double fy = fx;
    const double cx = img_width / 2.0;
    const double cy = img_height / 2.0;
    gtsam::Cal3_S2::shared_ptr K(new gtsam::Cal3_S2(fx, fy, 0, cx, cy));

    Backend backend(std::make_shared<FeatureManager>(), *K);
    test_scene.add_rand_cameras_face_origin(7, 2.0, 6.0, *K);

    std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras = test_scene.get_cameras();
    std::vector<Eigen::Isometry3d> camera_poses;
    camera_poses.reserve(cameras.size());
    for (const gtsam::PinholeCamera<gtsam::Cal3_S2> &T_world_cam : cameras) {
        camera_poses.emplace_back(T_world_cam.pose().matrix());
    }

    backend.add_prior_factor(gtsam::Symbol(backend.camera_symbol_char, 0), camera_poses.front(),
                             backend.get_pose_noise());
    for (int i = 1; i < camera_poses.size(); i++) {
        backend.add_factor_GPS(gtsam::Symbol(backend.camera_symbol_char, i), camera_poses[i],
                               backend.get_gps_noise());

        std::pair<gtsam::PinholeCamera<T>, cv::Size> cam_and_img_sizes[2] = {
            std::pair<gtsam::PinholeCamera<T>, cv::Size>(cameras[i - 1], img_size),
            std::pair<gtsam::PinholeCamera<T>, cv::Size>(cameras[i], img_size)};
        std::vector<std::pair<SpatialTestScene::ProjectedPoint, SpatialTestScene::ProjectedPoint>>
            lmks = test_scene.get_corresponding_pixels(cam_and_img_sizes);
        for (const &std::pair < SpatialTestScene::ProjectedPoint,
             SpatialTestScene::ProjectedPoint >> lmk : lmks) {
            // Backend::Landmark()
        }
    }

    geometry::viz_scene(camera_poses, test_scene.get_points());
}
}  // namespace robot::experimental::learn_descriptors