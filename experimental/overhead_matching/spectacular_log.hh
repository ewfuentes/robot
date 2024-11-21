
#pragma once

#include <filesystem>
#include <optional>

#include "Eigen/Core"
#include "common/math/cubic_hermite_spline.hh"
#include "common/time/robot_time.hh"
#include "opencv2/core/mat.hpp"
#include "opencv2/videoio.hpp"

namespace robot::experimental::overhead_matching {

struct FrameCalibration {
    Eigen::Vector2d focal_length;
    Eigen::Vector2d principal_point;

    std::optional<double> exposure_time_s;
    std::optional<double> depth_scale;
};

struct FrameGroup {
    time::RobotTimestamp time_of_validity;
    cv::Mat rgb_frame;
    cv::Mat depth_frame;

    FrameCalibration rgb_calibration;
    FrameCalibration depth_calibration;
};

struct ImuSample {
    time::RobotTimestamp time_of_validity;
    Eigen::Vector3d accel_mpss;
    Eigen::Vector3d gyro_radps;
};

namespace detail {
struct FrameInfo {
    double time_of_validity_s;
    int frame_number;
    std::array<FrameCalibration, 2> calibration;
};
}  // namespace detail

class SpectacularLog {
   public:
    explicit SpectacularLog(const std::filesystem::path &log_path);

    std::optional<ImuSample> get_imu_sample(const time::RobotTimestamp &t) const;
    std::optional<FrameGroup> get_frame(const int frame_id) const;

    const math::CubicHermiteSpline<Eigen::Vector3d> &gyro_spline() const { return gyro_spline_; }
    const math::CubicHermiteSpline<Eigen::Vector3d> &accel_spline() const { return accel_spline_; }

    time::RobotTimestamp min_imu_time() const;
    time::RobotTimestamp max_imu_time() const;

    time::RobotTimestamp min_frame_time() const;
    time::RobotTimestamp max_frame_time() const;

    int num_frames() const;

   private:
    std::filesystem::path log_path_;
    math::CubicHermiteSpline<Eigen::Vector3d> gyro_spline_;
    math::CubicHermiteSpline<Eigen::Vector3d> accel_spline_;
    std::vector<detail::FrameInfo> frame_info_;
    mutable std::unique_ptr<cv::VideoCapture> video_;
};
}  // namespace robot::experimental::overhead_matching
