#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <sstream>
#include <string>

#include "Eigen/Core"

namespace robot::experimental::learn_descriptors {
struct GPSData {
    struct Uncertainty {
        double sigma_latitude_m;
        double sigma_longitude_m;
        double sigma_altitude_m;
        double orientation_deg;
        double rms_range_error_m;

        // diagonal latitude, longitude, altitude covariance in meters squared
        Eigen::Matrix3d to_LLA_covariance() const {
            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            cov(0, 0) = sigma_latitude_m * sigma_latitude_m;
            cov(1, 1) = sigma_longitude_m * sigma_longitude_m;
            cov(2, 2) = sigma_altitude_m * sigma_altitude_m;
            return cov;
        }

        // convert lat/lon covariance to ENU
        Eigen::Matrix3d to_ENU_covariance() const {
            static constexpr double DEG2RAD = M_PI / 180.0;
            static constexpr double ORIENT_OFFSET_DEG = 90.0;  // CW‑from‑North → CCW‑from‑East

            Eigen::Matrix2d D;
            D << sigma_latitude_m * sigma_latitude_m, 0.0, 0.0,
                sigma_longitude_m * sigma_longitude_m;

            // rotate that into ENU axes
            double theta = (ORIENT_OFFSET_DEG - orientation_deg) * DEG2RAD;
            Eigen::Matrix2d R;
            R << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);

            Eigen::Matrix2d horz = R * D * R.transpose();

            Eigen::Matrix3d enu = Eigen::Matrix3d::Zero();
            enu.topLeftCorner<2, 2>() = horz;
            enu(2, 2) = sigma_altitude_m * sigma_altitude_m;
            return enu;
        }

        std::string to_string() const {
            std::stringstream ss;
            ss << "Uncertainty:";
            ss << "\n\tsigma_lat_deg: " << sigma_latitude_m;
            ss << "\n\tsigma_lon_deg: " << sigma_longitude_m;
            ss << "\n\tsigma_altitude_m: " << sigma_altitude_m;
            ss << "\n\torientation_deg: " << orientation_deg;
            ss << "\n\trms_range_error_m: " << rms_range_error_m;
            return ss.str();
        }
    };
    size_t seq;  // time in nanoseconds, may differ from image capture time
    double latitude;
    double longitude;
    std::optional<double> altitude;  // meters above sea level
    std::optional<Uncertainty> uncertainty;
    std::string to_string() const {
        std::stringstream ss;
        ss << "GPS Data: ";
        ss << "\n\tseq: " << seq;
        ss << "\n\tlla: " << latitude << ", " << longitude << ", "
           << (altitude ? std::to_string(*altitude) : std::string("N/A"));
        ss << "\n" << (uncertainty ? uncertainty->to_string() : "Uncertainty: N/A");
        return ss.str();
    }
};
}  // namespace robot::experimental::learn_descriptors