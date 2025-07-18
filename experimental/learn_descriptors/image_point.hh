#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "camera_calibration.hh"
#include "common/liegroups/se3.hh"
#include "experimental/learn_descriptors/camera_calibration.hh"

namespace robot::experimental::learn_descriptors {
struct ImagePoint {
    virtual ~ImagePoint() = default;

    size_t id;
    size_t seq;  // for time. TODO: make a time struct
    std::shared_ptr<CameraCalibrationFisheye> K;
    void set_cam_in_world(const Eigen::Vector3d& cam_in_world) { cam_in_world_ = cam_in_world; };
    virtual std::optional<Eigen::Isometry3d> world_from_cam_ground_truth() const {
        return std::nullopt;
    };
    virtual std::optional<Eigen::Vector3d> cam_in_world() const { return cam_in_world_; };
    virtual std::optional<Eigen::Matrix3d> translation_covariance_in_cam() const {
        return std::nullopt;
    };
    virtual std::string to_string() const {
        auto vec3d_to_str = [](const Eigen::Vector3d& vec3d) {
            std::stringstream ss;
            ss << "[" << vec3d.x() << ", " << vec3d.y() << ", " << vec3d.z() << "]";
            return ss.str();
        };
        auto isom3d_to_str = [&vec3d_to_str](const Eigen::Isometry3d& isom3d) {
            const Eigen::Vector3d& t = isom3d.translation();
            const Eigen::Quaterniond r(isom3d.rotation());
            std::stringstream ss;
            ss << "\t\tt: " << vec3d_to_str(t) << "\n";
            ss << "\t\tq: [" << r.w() << ", " << r.x() << ", " << r.y() << ", " << r.z() << "]";
            return ss.str();
        };
        std::stringstream ss;
        ss << "Image Point " << id << ":\n";
        ss << "\tworld_from_cam_ground_truth: ";
        if (world_from_cam_ground_truth()) {
            ss << "\n" << isom3d_to_str(*world_from_cam_ground_truth());
        } else {
            ss << "N/A";
        }
        ss << "\n\tcam_in_world: ";
        if (cam_in_world()) {
            ss << vec3d_to_str(*cam_in_world());
        } else {
            ss << "N/A";
        }
        ss << "\n\ttranslation_covariance_in_cam: ";
        if (translation_covariance_in_cam()) {
            ss << *translation_covariance_in_cam();
        } else {
            ss << "N/A";
        }
        return ss.str();
    }

   private:
    std::optional<Eigen::Vector3d> cam_in_world_;
};
}  // namespace robot::experimental::learn_descriptors