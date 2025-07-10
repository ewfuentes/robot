#pragma once

#include <cmath>
#include <cstddef>
#include <optional>

#include "Eigen/Core"

namespace robot::experimental::learn_descriptors {
struct GPSData {
    struct Uncertainty {
        double sigma_lat_mitude;
        double sigma_longitude;
        double sigma_altitude;
        double orientation_deg;
        double rms_range_error_m;

        // diagonal latitude, longitude, altitude covariance in meters squared
        Eigen::Matrix3d to_LLA_covariance() const {
            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            cov(0, 0) = sigma_lat_mitude * sigma_lat_mitude;
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
}  // namespace robot::experimental::learn_descriptors