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
        double longitude;
        double latitude;
        std::optional<double> altitude;  // meters above sea level
    };
    size_t id;   // idx for DB
    size_t seq;  // time in nanoseconds (also the name of image)
    std::optional<liegroups::SE3> reference;
    std::optional<liegroups::SE3> vio_solution;
    std::optional<GPSData> gps_gcs;

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
        ss << "\reference: ";
        if (reference) {
            ss << "\n" << se3_to_str(*reference);
        } else {
            ss << "N/A";
        }
        ss << "\n\tvio_solution: ";
        if (vio_solution) {
            ss << "\n" << se3_to_str(*vio_solution);
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