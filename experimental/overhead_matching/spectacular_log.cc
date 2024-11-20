
#include "experimental/overhead_matching/spectacular_log.hh"

#include <array>
#include <filesystem>
#include <fstream>
#include <string>

#include "Eigen/Core"
#include "common/check.hh"
#include "common/math/cubic_hermite_spline.hh"
#include "common/time/robot_time.hh"
#include "nlohmann/json.hpp"

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

struct FrameCalibration {
    Eigen::Vector2d focal_length;
    Eigen::Vector2d principal_point;

    std::optional<double> exposure_time_s;
    std::optional<double> depth_scale;
};

struct FrameGroup {
    double time_of_validity_s;
    int frame_number;
    std::array<FrameCalibration, 2> calibration;
};

struct LogData {
    std::vector<AccelMeasurement> accel;
    std::vector<GyroMeasurement> gyro;
    std::vector<FrameGroup> frames;
};

LogData read_jsonl(const fs::path &file_path) {
    CHECK(fs::exists(file_path), "jsonl path does not exist", file_path);
    std::ifstream in_file(file_path);
    CHECK(in_file.is_open(), "Could not open file", file_path);

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
                CHECK(frame["colorFormat"] == "rgb");

                calibration[0].focal_length = {calib_info["focalLengthX"].get<double>(),
                                               calib_info["focalLengthY"].get<double>()};
                calibration[0].principal_point = {calib_info["principalPointX"].get<double>(),
                                                  calib_info["principalPointY"].get<double>()};
                calibration[0].exposure_time_s = frame["exposureTimeSeconds"].get<double>();
            }
            {
                const auto &frame = j["frames"].at(1);
                const auto &calib_info = frame["calibration"];
                CHECK(frame["colorFormat"] == "gray");

                calibration[1].focal_length = {calib_info["focalLengthX"].get<double>(),
                                               calib_info["focalLengthY"].get<double>()};
                calibration[1].principal_point = {calib_info["principalPointX"].get<double>(),
                                                  calib_info["principalPointY"].get<double>()};
                calibration[1].depth_scale = frame["depthScale"].get<double>();
            }

            out.frames.push_back(FrameGroup{
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

}  // namespace

SpectacularLog::SpectacularLog(const fs::path &path) {
    const auto data_path = path / "data.jsonl";
    const LogData log_data = read_jsonl(data_path);
    gyro_spline_ =
        make_spline(log_data.gyro, [](const auto &frame) { return frame.angular_vel_radps; });
    accel_spline_ = make_spline(log_data.accel, [](const auto &frame) { return frame.accel_mpss; });
}
}  // namespace robot::experimental::overhead_matching
