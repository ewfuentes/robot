#pragma once

#include <cstddef>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <optional>
#include <sstream>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "common/liegroups/se3.hh"
#include "common/time/robot_time.hh"

namespace robot::experimental::learn_descriptors {
struct ImagePoint {
    struct GPSData {
        struct Uncertainty {
            double sigma_latitude;
            double sigma_longitude;
            double sigma_altitude;
            double orientation_deg;
            double rms_range_error;

            // diagonal latitude, longitude, altitude covariance in meters squared
            Eigen::Matrix3d to_LLA_covariance() const {
                Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                cov(0, 0) = sigma_latitude * sigma_latitude;
                cov(1, 1) = sigma_longitude * sigma_longitude;
                cov(2, 2) = sigma_altitude * sigma_altitude;
                return cov;
            }

            // Convert lat/lon covariance to ENU at given latitude
            Eigen::Matrix3d to_ENU_covariance(double lat_deg) const {
                double lat_rad = lat_deg * M_PI / 180.0;

                // meters per degree scaling
                double kLat =
                    111132.92 - 559.82 * std::cos(2 * lat_rad) + 1.175 * std::cos(4 * lat_rad);
                double kLon = 111412.84 * std::cos(lat_rad) - 93.5 * std::cos(3 * lat_rad);

                Eigen::Matrix3d lla_cov = to_LLA_covariance();

                // rotate lat/lon covariance (upper-left 2x2) using orientation
                double theta = orientation_deg * M_PI / 180.0;
                Eigen::Matrix2d R;
                R << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);

                Eigen::Matrix2d D = lla_cov.topLeftCorner<2, 2>();
                Eigen::Matrix2d rotated_latlon_cov = R * D * R.transpose();

                // apply linear scaling to convert degrees to meters in EN
                Eigen::Matrix2d J;
                J << kLat, 0, 0, kLon;

                Eigen::Matrix2d en_cov = J * rotated_latlon_cov * J.transpose();

                Eigen::Matrix3d enu_cov = Eigen::Matrix3d::Zero();
                enu_cov.topLeftCorner<2, 2>() = en_cov;
                enu_cov(2, 2) = lla_cov(2, 2);  // alt variance unchanged

                return enu_cov;
            }
        };
        size_t seq;  // time in nanoseconds, may differ from image capturetime
        double latitude;
        double longitude;
        std::optional<double> altitude;  // meters above sea level
        std::optional<Uncertainty> uncertainty;
    };
    size_t id;   // idx for DB
    size_t seq;  // time in nanoseconds of image capture (also the name of image)
    std::optional<liegroups::SE3>
        AS_w_from_gnss_cam;  // globally opimized arbitrary scale visual world (gnss + VIO + loop
                             // closure + etc.) from cam
    std::optional<liegroups::SE3> AS_w_from_vio_cam;  // arbitrary scale world from vio result cam
    std::optional<GPSData> gps_gcs;  // raw gps measurement in gcs (global cordinate system)

    const std::string to_string() const {
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
            ss << "\n\t\t" << gps_gcs->latitude << "\t" << gps_gcs->longitude << "\t";
            if (gps_gcs->altitude) {
                ss << *(gps_gcs->altitude);
            } else {
                ss << "alt N/A";
            }
            ss << "\n\t\tsigma: ";
            if (gps_gcs->uncertainty) {
                ss << gps_gcs->uncertainty->sigma_latitude << "\t"
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
typedef std::vector<ImagePoint> ImagePointVector;
}  // namespace robot::experimental::learn_descriptors