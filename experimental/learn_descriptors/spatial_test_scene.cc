#include "experimental/learn_descriptors/spatial_test_scene.hh"

#include <random>

namespace robot::experimental::learn_descriptors {
template <typename T>
std::vector<SpatialTestScene::ProjectedPoint> SpatialTestScene::get_projected_pixels(
    const gtsam::PinholeCamera<T> &camera, const cv::Size &img_size) {
    std::vector<ProjectedPoint> pixels;
    for (size_t i = 0; i < points_.size(); i++) {
        const Eigen::Vector3d pt = points_[i];
        Eigen::Vector2d pixel = camera.project(pt);
        if (ProjectionHelper::pixel_in_range(pixel, img_size.width, img_size.height)) {
            ProjectedPoint projected_pt(i, pixel);
            pixels.push_back(projected_pt);
        }
    }
    return pixels;
};

template <typename T>
std::vector<std::pair<SpatialTestScene::ProjectedPoint, SpatialTestScene::ProjectedPoint>>
SpatialTestScene::get_corresponding_pixels(
    const std::vector<std::pair<gtsam::PinholeCamera<T>, cv::Size>> &cam_and_img_size) {
    std::vector<std::pair<ProjectedPoint, ProjectedPoint>> corresponding_pixels;
    for (size_t i = 0; i < points_.size(); i++) {
        const Eigen::Vector3d pt = points_[i];
        Eigen::Vector2d pixel_cam0 = cam_and_img_size[0].first.project(pt);
        Eigen::Vector2d pixel_cam1 = cam_and_img_size[1].first.project(pt);
        if (ProjectionHelper::pixel_in_range(pixel_cam0, cam_and_img_size[0].second.width,
                                             cam_and_img_size[0].second.height) &&
            ProjectionHelper::pixel_in_range(pixel_cam1, cam_and_img_size[1].second.width,
                                             cam_and_img_size[1].second.height)) {
            corresponding_pixels.emplace_back(ProjectedPoint(i, pixel_cam0),
                                              ProjectedPoint(i, pixel_cam1));
        }
    }
    return corresponding_pixels;
};

void SpatialTestScene::add_point(Eigen::Vector3d point) { points_.push_back(point); };

void SpatialTestScene::add_points(std::vector<Eigen::Vector3d> points) {
    points_.insert(points_.end(), points.begin(), points.end());
}

void SpatialTestScene::add_camera(gtsam::PinholeCamera<gtsam::Cal3_S2> camera) {
    cameras_.push_back(camera);
}

void SpatialTestScene::add_cameras(std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras) {
    cameras_.insert(cameras_.end(), cameras.begin(), cameras.end());
}

void SpatialTestScene::add_rand_cameras_face_origin(int num_cameras, double min_radius_origin,
                                                    double max_radius_origin,
                                                    const gtsam::Cal3_S2 &K) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<double> dist_radius(min_radius_origin, max_radius_origin);
    std::uniform_real_distribution<double> dist_omega(0.0, 2 * M_PI);

    for (int i = 0; i < num_cameras; i++) {
        double angle_z = dist_radius(gen);
        Eigen::Isometry3d R_z;
        R_z.linear() = Eigen::AngleAxisd(angle_z, Eigen::Vector3d::UnitZ()).matrix();
        double angle_x = dist_radius(gen);
        Eigen::Isometry3d R_y;
        R_y.linear() = Eigen::AngleAxisd(angle_x, Eigen::Vector3d::UnitY()).matrix();

        double radius = dist_radius(gen);
        Eigen::Isometry3d T_world_cam;
        T_world_cam.translation() = Eigen::Vector3d(radius, 0, 0);
        T_world_cam.linear() = (Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitY()) *
                                Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()))
                                   .matrix();
        T_world_cam = R_z * R_y * T_world_cam;
        cameras_.push_back(gtsam::PinholeCamera(gtsam::Pose3(T_world_cam.matrix()), K));
    }
}
}  // namespace robot::experimental::learn_descriptors