#include "experimental/learn_descriptors/spatial_test_scene.hh"

#include <random>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/check.hh"
#include "gtsam/geometry/Cal3_S2.h"
#include "gtsam/geometry/PinholeCamera.h"
#include "opencv2/opencv.hpp"

Eigen::Isometry3d isometry_from_vector(const Eigen::VectorXd &pose_vector) {
    ROBOT_CHECK(pose_vector.size() == 6);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    if (pose_vector.tail<3>().norm() > 1e-6) {
        T.linear() = Eigen::AngleAxisd(pose_vector.tail<3>().norm(), pose_vector.tail<3>())
                         .toRotationMatrix();
    }
    T.translation() = pose_vector.head<3>();
    return T;
}

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

void SpatialTestScene::add_point(Eigen::Vector3d point, std::optional<Noise> point_noise) {
    points_groundtruth_.push_back(point);
    if (point_noise) {
        ROBOT_CHECK(point_noise->dim() == 3);
        points_.push_back(point + point_noise->sample());
    } else {
        points_.push_back(point);
    }
};

void SpatialTestScene::add_points(std::vector<Eigen::Vector3d> points,
                                  std::optional<Noise> point_noise) {
    for (const Eigen::Vector3d &point : points) {
        points_groundtruth_.push_back(point);
        if (point_noise) {
            ROBOT_CHECK(point_noise->dim() == 3);
            points_.push_back(point + point_noise->sample());
        } else {
            points_.push_back(point);
        }
    }
}

void SpatialTestScene::add_camera(gtsam::PinholeCamera<gtsam::Cal3_S2> camera,
                                  std::optional<Noise> pose_noise) {
    cameras_groundtruth_.push_back(camera);
    if (pose_noise) {
        cameras_.push_back(gtsam::PinholeCamera<gtsam::Cal3_S2>(
            camera.pose() * gtsam::Pose3(isometry_from_vector(pose_noise->sample()).matrix()),
            camera.calibration()));
    } else {
        cameras_.push_back(camera);
    }
}

void SpatialTestScene::add_cameras(std::vector<gtsam::PinholeCamera<gtsam::Cal3_S2>> cameras,
                                   std::optional<Noise> pose_noise) {
    for (const gtsam::PinholeCamera<gtsam::Cal3_S2> &cam : cameras) {
        cameras_groundtruth_.push_back(cam);
        if (pose_noise) {
            cameras_.push_back(gtsam::PinholeCamera<gtsam::Cal3_S2>(
                cam.pose() * gtsam::Pose3(isometry_from_vector(pose_noise->sample()).matrix()),
                cam.calibration()));
        } else {
            cameras_.push_back(cam);
        }
    }
}

void SpatialTestScene::add_rand_cameras_face_origin(int num_cameras, double min_radius_origin,
                                                    double max_radius_origin,
                                                    std::optional<Noise> pose_noise,
                                                    const gtsam::Cal3_S2 &K) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<double> dist_radius(min_radius_origin, max_radius_origin);
    std::uniform_real_distribution<double> dist_omega(0.0, 2 * M_PI);

    const Eigen::Isometry3d world_from_cam_default =
        Eigen::Translation3d(1.0, 0, 0) * Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ());
    for (int i = 0; i < num_cameras; i++) {
        double angle_z = dist_omega(gen);
        Eigen::Isometry3d longitude_from_world = Eigen::Isometry3d::Identity();
        longitude_from_world.linear() =
            Eigen::AngleAxisd(angle_z, Eigen::Vector3d::UnitZ()).matrix();
        double angle_x = dist_omega(gen);
        Eigen::Isometry3d latitude_from_world = Eigen::Isometry3d::Identity();
        latitude_from_world.linear() =
            Eigen::AngleAxisd(angle_x, Eigen::Vector3d::UnitY()).matrix();

        double radius = dist_radius(gen);
        Eigen::Isometry3d world_from_cam(world_from_cam_default);
        world_from_cam.translation() *= radius;
        world_from_cam = longitude_from_world * latitude_from_world * world_from_cam;
        cameras_groundtruth_.push_back(
            gtsam::PinholeCamera(gtsam::Pose3(world_from_cam.matrix()), K));
        if (pose_noise) {
            cameras_.push_back(gtsam::PinholeCamera(
                gtsam::Pose3(
                    (world_from_cam * isometry_from_vector(pose_noise->sample())).matrix()),
                K));
        } else {
            cameras_.push_back(gtsam::PinholeCamera(gtsam::Pose3(world_from_cam.matrix()), K));
        }
    }
}
}  // namespace robot::experimental::learn_descriptors