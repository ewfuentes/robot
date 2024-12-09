
#include "experimental/overhead_matching/spectacular_log.hh"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "Eigen/Core"
#include "common/argument_wrapper.hh"
#include "common/check.hh"
#include "common/math/cubic_hermite_spline.hh"
#include "common/time/robot_time.hh"
#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "opencv2/imgcodecs.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace robot::experimental::overhead_matching {

namespace {

struct AccelMeasurement {
    double time_of_validity_s;
    Eigen::Vector3d accel_mpss;
};

struct GyroMeasurement {
    double time_of_validity_s;
    Eigen::Vector3d angular_vel_radps;
};

struct LogData {
    std::vector<AccelMeasurement> accel;
    std::vector<GyroMeasurement> gyro;
    std::vector<detail::FrameInfo> frames;
};

LogData read_jsonl(const fs::path &file_path) {
    ROBOT_CHECK(fs::exists(file_path), "jsonl path does not exist", file_path);
    std::ifstream in_file(file_path);
    ROBOT_CHECK(in_file.is_open(), "Could not open file", file_path);

    LogData out;
    for (std::string line; std::getline(in_file, line);) {
        const auto j = json::parse(line);
        const double time_of_validity_s = j["time"].get<double>();
        if (j.contains("sensor")) {
            const auto &sensor_data = j["sensor"];
            const auto &v = sensor_data["values"];
            if (sensor_data["type"] == "accelerometer") {
                out.accel.emplace_back(AccelMeasurement{
                    .time_of_validity_s = time_of_validity_s,
                    .accel_mpss = Eigen::Vector3d{v.at(0).get<double>(), v.at(1).get<double>(),
                                                  v.at(2).get<double>()},
                });
            } else if (sensor_data["type"] == "gyroscope") {
                out.gyro.emplace_back(GyroMeasurement{
                    .time_of_validity_s = time_of_validity_s,
                    .angular_vel_radps =
                        Eigen::Vector3d{v.at(0).get<double>(), v.at(1).get<double>(),
                                        v.at(2).get<double>()},
                });
            }
        } else if (j.contains("frames")) {
            const int frame_number = j["number"].get<int>();
            std::array<FrameCalibration, 2> calibration;
            {
                const auto &frame = j["frames"].at(0);
                const auto &calib_info = frame["calibration"];
                ROBOT_CHECK(frame["colorFormat"] == "rgb");

                calibration[0].focal_length = {calib_info["focalLengthX"].get<double>(),
                                               calib_info["focalLengthY"].get<double>()};
                calibration[0].principal_point = {calib_info["principalPointX"].get<double>(),
                                                  calib_info["principalPointY"].get<double>()};
                calibration[0].exposure_time_s = frame["exposureTimeSeconds"].get<double>();
            }
            {
                const auto &frame = j["frames"].at(1);
                const auto &calib_info = frame["calibration"];
                ROBOT_CHECK(frame["colorFormat"] == "gray");

                calibration[1].focal_length = {calib_info["focalLengthX"].get<double>(),
                                               calib_info["focalLengthY"].get<double>()};
                calibration[1].principal_point = {calib_info["principalPointX"].get<double>(),
                                                  calib_info["principalPointY"].get<double>()};
                calibration[1].depth_scale = frame["depthScale"].get<double>();
            }

            out.frames.push_back(detail::FrameInfo{
                .time_of_validity_s = time_of_validity_s,
                .frame_number = frame_number,
                .calibration = calibration,
            });
        }
    }
    return out;
}

template <typename T>
math::CubicHermiteSpline<Eigen::Vector3d> make_spline(const std::vector<T> &frames,
                                                      const auto &accessor) {
    std::vector<double> ts;
    std::vector<Eigen::Vector3d> xs;
    ts.reserve(frames.size());
    xs.reserve(frames.size());

    for (const auto &frame : frames) {
        ts.push_back(frame.time_of_validity_s);
        xs.push_back(accessor(frame));
    }
    return math::CubicHermiteSpline(ts, xs);
}

void seek_to_frame(const int frame_id, InOut<cv::VideoCapture> video) {
    const int current_frame_id = video->get(cv::CAP_PROP_POS_FRAMES);

    if (current_frame_id > frame_id) {
        video->set(cv::CAP_PROP_POS_FRAMES, 0);
    }

    while (video->get(cv::CAP_PROP_POS_FRAMES) != frame_id) {
        video->grab();
    }
}

}  // namespace

SpectacularLog::SpectacularLog(const fs::path &path) : log_path_(path) {
    const auto data_path = log_path_ / "data.jsonl";
    LogData log_data = read_jsonl(data_path);
    gyro_spline_ =
        make_spline(log_data.gyro, [](const auto &frame) { return frame.angular_vel_radps; });
    accel_spline_ = make_spline(log_data.accel, [](const auto &frame) { return frame.accel_mpss; });
    frame_info_ = std::move(log_data.frames);
}

std::optional<ImuSample> SpectacularLog::get_imu_sample(const time::RobotTimestamp &t) const {
    if (t < min_imu_time() || t > max_imu_time()) {
        return std::nullopt;
    }

    return {{
        .time_of_validity = t,
        .accel_mpss = accel_spline_(std::chrono::duration<double>(t.time_since_epoch()).count()),
        .gyro_radps = gyro_spline_(std::chrono::duration<double>(t.time_since_epoch()).count()),
    }};
}

std::optional<FrameGroup> SpectacularLog::get_frame(const int frame_id) const {
    if (frame_id < 0 || frame_id >= static_cast<int>(frame_info_.size())) {
        return std::nullopt;
    }

    const auto &frame_info = frame_info_.at(frame_id);

    // Read the desired rgb frame
    if (video_ == nullptr) {
        const fs::path color_path = log_path_ / "data.mov";
        video_ = std::make_unique<cv::VideoCapture>(color_path.string(), cv::CAP_FFMPEG);
    }
    cv::Mat bgr_frame;
    seek_to_frame(frame_info.frame_number, make_in_out(*video_));
    video_->read(bgr_frame);

    // Read the desired depth frame
    const fs::path depth_frame_path =
        log_path_ / "frames2" / fmt::format("{:08d}.png", frame_info.frame_number);

    // TODO: change to 32FC1 depth in meters
    cv::Mat depth_frame = cv::imread(depth_frame_path.string(), cv::IMREAD_GRAYSCALE);
    // temp solution:
    depth_frame.convertTo(depth_frame, CV_32FC1);

    return {{
        .time_of_validity =
            time::as_duration(frame_info.time_of_validity_s) + time::RobotTimestamp(),
        .bgr_frame = std::move(bgr_frame),
        .depth_frame = std::move(depth_frame),
        .color_calibration = frame_info.calibration.at(0),
        .depth_calibration = frame_info.calibration.at(1),
    }};
}

time::RobotTimestamp SpectacularLog::min_imu_time() const {
    const double later_time_s = std::max(gyro_spline_.min_time(), accel_spline_.min_time());
    return time::as_duration(later_time_s + 1e-6) + time::RobotTimestamp();
}

time::RobotTimestamp SpectacularLog::max_imu_time() const {
    const double earlier_time_s = std::min(gyro_spline_.max_time(), accel_spline_.max_time());
    return time::as_duration(earlier_time_s - 1e-6) + time::RobotTimestamp();
}

time::RobotTimestamp SpectacularLog::min_frame_time() const {
    return time::as_duration(frame_info_.front().time_of_validity_s) + time::RobotTimestamp();
}

time::RobotTimestamp SpectacularLog::max_frame_time() const {
    return time::as_duration(frame_info_.back().time_of_validity_s) + time::RobotTimestamp();
}

int SpectacularLog::num_frames() const { return frame_info_.size(); }
}  // namespace robot::experimental::overhead_matching
