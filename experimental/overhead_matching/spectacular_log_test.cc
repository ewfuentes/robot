
#include "experimental/overhead_matching/spectacular_log.hh"

#include <iostream>
#include <sstream>

#include "common/matplotlib.hh"
#include "fmt/format.h"
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"
#include "common/video.hh"

namespace robot::experimental::overhead_matching {

std::ostream &operator<<(std::ostream &out, const time::RobotTimestamp &t) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(9)
       << std::chrono::duration<double>(t.time_since_epoch()).count();
    out << ss.str();
    return out;
}

TEST(SpectacularLogTest, happy_case) {
    // Setup
    const std::filesystem::path log_path(
        "external/spectacular_log_snippet/recording_2024-11-21_13-36-30");
    SpectacularLog log(log_path);
    // Action

    std::cout << "imu time: (" << log.min_imu_time() << ", " << log.max_imu_time() << ")"
              << " frame time: (" << log.min_frame_time() << ", " << log.max_frame_time() << ")"
              << std::endl;

    std::vector<double> ts;
    std::vector<double> axs;
    std::vector<double> ays;
    std::vector<double> azs;
    std::vector<double> gxs;
    std::vector<double> gys;
    std::vector<double> gzs;
    std::cout << "IMU Samples" << std::endl;
    for (time::RobotTimestamp t = log.min_imu_time();
         t < std::min(log.min_imu_time() + time::as_duration(20.0), log.max_imu_time());
         t += time::as_duration(0.1)) {
        const auto sample = log.get_imu_sample(t);
        if (sample.has_value()) {
            ts.push_back(std::chrono::duration<double>(t.time_since_epoch()).count());
            axs.push_back(sample->accel_mpss.x());
            ays.push_back(sample->accel_mpss.y());
            azs.push_back(sample->accel_mpss.z());
            gxs.push_back(sample->gyro_radps.x());
            gys.push_back(sample->gyro_radps.y());
            gzs.push_back(sample->gyro_radps.z());
            std::cout << "t: " << sample->time_of_validity
                      << " accel: " << sample->accel_mpss.transpose()
                      << " gyro: " << sample->gyro_radps.transpose() << std::endl;
        }
    }

    if (false) {
        const bool should_block = false;
        plot(
            {
                {ts, axs, "ax"},
                {ts, ays, "ay"},
                {ts, azs, "az"},
                {ts, gxs, "gx"},
                {ts, gys, "gy"},
                {ts, gzs, "gz"},
            },
            should_block);
    }

    cv::VideoCapture video(log_path / "data.mov", cv::CAP_FFMPEG);
    constexpr int FRAME_SKIP = 50;
    cv::Mat expected_frame;
    for (int frame_id = 0; frame_id < log.num_frames(); frame_id += FRAME_SKIP) {
        video.read(expected_frame);

        const auto frame = log.get_frame(frame_id).value();

        const std::filesystem::path depth_path(log_path /
                                               fmt::format("frames2/{:08d}.png", frame_id));
        const cv::Mat depth_frame = cv::imread(depth_path, cv::IMREAD_GRAYSCALE);

        EXPECT_TRUE(common::images_equal(expected_frame, frame.bgr_frame));
        EXPECT_TRUE(common::images_equal(depth_frame, frame.depth_frame));

        for (int i = 0; i < FRAME_SKIP - 1; i++) {
            video.grab();
        }
    }
}
}  // namespace robot::experimental::overhead_matching
