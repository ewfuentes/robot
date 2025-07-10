#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/liegroups/se3.hh"
#include "experimental/learn_descriptors/four_seasons_transforms.hh"
#include "experimental/learn_descriptors/gps_data.hh"
#include "experimental/learn_descriptors/image_point.hh"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
class FourSeasonsParser;
struct ImagePointFourSeasons : ImagePoint {
    virtual ~ImagePointFourSeasons();

    size_t id;   // idx for DB
    size_t seq;  // time in nanoseconds of image capture (also the name of image)
    std::optional<liegroups::SE3>
        AS_w_from_gnss_cam;  // globally opimized arbitrary scale visual world (gnss + VIO + loop
                             // closure + etc.) from cam
    std::optional<liegroups::SE3> AS_w_from_vio_cam;  // arbitrary scale world from vio result cam
    std::optional<GPSData> gps_gcs;  // raw gps measurement in gcs (global cordinate system)
    std::shared_ptr<FourSeasonsTransforms::StaticTransforms> shared_static_transforms;

    std::optional<Eigen::Isometry3d> world_from_cam_ground_truth() const override;
    std::optional<Eigen::Vector3d> cam_in_world() const override;
    std::optional<Eigen::Matrix3d> translation_covariance_in_cam() const override;
    std::optional<Eigen::Matrix3d> gps_covariance_in_world() const;

    std::string to_string() const {
        auto se3_to_str = [](const liegroups::SE3& se3) {
            const Eigen::Vector3d& t = se3.translation();
            const Eigen::Quaterniond& r = se3.so3().unit_quaternion();
            std::stringstream ss;
            ss << "\t\tt: [" << t.x() << ", " << t.y() << ", " << t.z() << "]\n";
            ss << "\t\tq: [" << r.w() << ", " << r.x() << ", " << r.y() << ", " << r.z() << "]";
            return ss.str();
        };
        std::stringstream ss;
        ss << "Image Point " << id << ":\n";
        ss << "\tseq: " << seq << "\n";
        ss << "\tAS_w_from_gnss_cam: ";
        if (AS_w_from_gnss_cam) {
            ss << "\n" << se3_to_str(*AS_w_from_gnss_cam);
        } else {
            ss << "N/A";
        }
        ss << "\n\tAS_w_from_vio_cam: ";
        if (AS_w_from_vio_cam) {
            ss << "\n" << se3_to_str(*AS_w_from_vio_cam);
        } else {
            ss << "N/A";
        }
        ss << "\n\tgps_gcs: ";
        if (gps_gcs) {
            ss << "\n\t\tseq: " << gps_gcs->seq;
            ss << "\n\t\tval: " << gps_gcs->latitude << "\t" << gps_gcs->longitude << "\t";
            if (gps_gcs->altitude) {
                ss << *(gps_gcs->altitude);
            } else {
                ss << "alt N/A";
            }
            ss << "\n\t\tsigma: ";
            if (gps_gcs->uncertainty) {
                ss << gps_gcs->uncertainty->sigma_lat_mitude << "\t"
                   << gps_gcs->uncertainty->sigma_longitude << "\t"
                   << gps_gcs->uncertainty->sigma_altitude;
            } else {
                ss << "N/A";
            }
        } else {
            ss << "N/A";
        }
        return ss.str();
    }

    cv::Mat load_image(const std::filesystem::path& img_dir) const {
        const std::filesystem::path path(img_dir / (std::to_string(seq) + ".png"));
        return cv::imread(path);
    }
};
typedef std::vector<ImagePointFourSeasons> ImagePointFourSeasonsVector;
}  // namespace robot::experimental::learn_descriptors