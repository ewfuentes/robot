#pragma once

#include <cmath>
#include <cstddef>
#include <optional>

#include "Eigen/Core"

namespace robot::experimental::learn_descriptors {
struct GPSData {
    struct Uncertainty {
        double sigma_latitude_deg;
        double sigma_longitude_deg;
        double sigma_altitude_m;
        double orientation_deg;
        double rms_range_error_m;

        // diagonal latitude, longitude, altitude covariance in meters squared
        Eigen::Matrix3d to_LLA_covariance() const {
            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            cov(0, 0) = sigma_latitude_deg * sigma_latitude_deg;
            cov(1, 1) = sigma_longitude_deg * sigma_longitude_deg;
            cov(2, 2) = sigma_altitude_m * sigma_altitude_m;
            return cov;
        }

        // Convert lat/lon covariance to ENU at given latitude
        Eigen::Matrix3d to_ENU_covariance(double lat_deg) const {
            // –– Named conversion/constants ––
            static constexpr double DEG2RAD = M_PI / 180.0;
            static constexpr double ORIENTATION_OFFSET_DEG =
                90.0;  // for CW‐from‐North → CCW‐from‐East

            // WGS‑84 approximate “meters per degree” polynomial coefficients
            static constexpr double MERIDIONAL_A0 = 111132.92;
            static constexpr double MERIDIONAL_A2 = -559.82;
            static constexpr double MERIDIONAL_A4 = 1.175;
            static constexpr double PARALLEL_B0 = 111412.84;
            static constexpr double PARALLEL_B2 = -93.5;

            // 1) get LLA covariance in degrees²
            Eigen::Matrix3d lla_cov = to_LLA_covariance();

            // 2) convert latitude to radians
            double phi = lat_deg * DEG2RAD;

            // 3) compute meters/degree at this latitude
            double kLat = MERIDIONAL_A0 + MERIDIONAL_A2 * std::cos(2.0 * phi) +
                          MERIDIONAL_A4 * std::cos(4.0 * phi);
            double kLon = PARALLEL_B0 * std::cos(phi) + PARALLEL_B2 * std::cos(3.0 * phi);

            // 4) build the Jacobian for degree→meter scaling
            Eigen::Matrix2d J;
            J << kLat, 0.0, 0.0, kLon;

            // 5) scale the 2×2 lat/lon covariance into meters²
            Eigen::Matrix2d D = lla_cov.topLeftCorner<2, 2>();
            Eigen::Matrix2d cov_m = J * D * J.transpose();

            // 6) rotate that ellipse into ENU axes (θ = CCW from East)
            double theta = (ORIENTATION_OFFSET_DEG - orientation_deg) * DEG2RAD;
            Eigen::Matrix2d R;
            R << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);
            Eigen::Matrix2d horz_cov = R * cov_m * R.transpose();

            // 7) assemble full 3×3 ENU covariance
            Eigen::Matrix3d enu_cov = Eigen::Matrix3d::Zero();
            enu_cov.topLeftCorner<2, 2>() = horz_cov;
            enu_cov(2, 2) = lla_cov(2, 2);  // altitude variance already in m²

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