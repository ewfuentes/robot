#include "experimental/learn_descriptors/image_point_four_seasons.hh"

#include <optional>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/gps/frame_translation.hh"
#include "common/liegroups/se3.hh"
#include "experimental/learn_descriptors/four_seasons_transforms.hh"

namespace robot::experimental::learn_descriptors {
ImagePointFourSeasons::~ImagePointFourSeasons() = default;

std::optional<Eigen::Isometry3d> ImagePointFourSeasons::world_from_cam_ground_truth() const {
    if (!AS_w_from_gnss_cam) return std::nullopt;
    Eigen::Matrix4d scale_mat_reference = Eigen::Matrix4d::Identity();
    scale_mat_reference(0, 0) = scale_mat_reference(1, 1) = scale_mat_reference(2, 2) =
        shared_static_transforms->gnss_scale;
    Eigen::Isometry3d world_from_cam_ground_truth(
        (shared_static_transforms->S_from_AS.matrix() * scale_mat_reference *
         AS_w_from_gnss_cam->matrix())
            .matrix());
    return world_from_cam_ground_truth;
}

std::optional<Eigen::Vector3d> ImagePointFourSeasons::cam_in_world() const {
    if (!gps_gcs) return std::nullopt;
    const Eigen::Vector3d gps_in_ECEF(gps::ecef_from_lla(
        Eigen::Vector3d(gps_gcs->latitude, gps_gcs->longitude, *gps_gcs->altitude)));
    Eigen::Vector3d cam_in_world(shared_static_transforms->cam_from_imu *
                                 shared_static_transforms->gps_from_imu.inverse() *
                                 shared_static_transforms->w_from_gpsw *
                                 shared_static_transforms->e_from_gpsw.inverse() * gps_in_ECEF);
    return cam_in_world;
}

std::optional<Eigen::Matrix3d> ImagePointFourSeasons::translation_covariance_in_cam() const {
    return gps_covariance_in_world();
}

std::optional<Eigen::Matrix3d> ImagePointFourSeasons::gps_covariance_in_world() const {
    if (!gps_gcs || !gps_gcs->uncertainty) return std::nullopt;
    const Eigen::Matrix3d ENU_covariance(
        gps_gcs->uncertainty->to_ENU_covariance(gps_gcs->latitude));
    const Eigen::Matrix3d w_from_ENU(shared_static_transforms->w_from_gpsw.so3().matrix());
    const Eigen::Matrix3d w_covariance(w_from_ENU * ENU_covariance * w_from_ENU.transpose());
    return w_covariance;
}
}  // namespace robot::experimental::learn_descriptors