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
    size_t id;                          // idx for DB
    size_t seq;                         // time in nanoseconds (also the name of image)
    std::optional<liegroups::SE3> gps;  // UTM. Not sure why the dataset has transforms for GPS, not
                                        // sure if rotation or trans_z is reliable
    std::optional<liegroups::SE3> ground_truth;

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
        ss << "\tgps: ";
        if (gps) {
            ss << "\n" << se3_to_str(*gps);
        } else {
            ss << "N/A";
        }
        ss << "\n\tground_truth: ";
        if (ground_truth) {
            ss << "\n" << se3_to_str(*ground_truth);
        } else {
            ss << "N/A";
        }
        return ss.str();
    }

    cv::Mat load_image(const std::filesystem::path& img_dir) const {
        const std::filesystem::path path(img_dir / (std::to_string(seq) + ".png"));
        if (std::getenv("DEBUG")) {
            std::cout << "getting image at " << path << std::endl;
        }
        return cv::imread(path);
    }
};
typedef std::vector<ImagePoint> ImagePointVector;
}  // namespace robot::experimental::learn_descriptors