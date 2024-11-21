
#include "experimental/overhead_matching/spectacular_log.hh"

#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::overhead_matching {

bool images_equal(cv::Mat img1, cv::Mat img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        return false;
    }
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff = diff.reshape(1);
    return cv::countNonZero(diff) == 0;
}

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

    std::cout << "IMU Samples" << std::endl;
    for (time::RobotTimestamp t = log.min_imu_time();
         t < std::min(log.min_imu_time() + time::as_duration(5.0), log.max_imu_time());
         t += time::as_duration(0.1)) {
        const auto sample = log.get_imu_sample(t);
        if (sample.has_value()) {
            std::cout << "t: " << sample->time_of_validity
                      << " accel: " << sample->accel_mpss.transpose()
                      << " gyro: " << sample->gyro_radps.transpose() << std::endl;
        }
    }

    cv::VideoCapture video(log_path / "data.mov", cv::CAP_FFMPEG);
    constexpr int FRAME_SKIP = 50;
    cv::Mat expected_frame;
    for (int frame_id = 0; frame_id < log.num_frames(); frame_id += FRAME_SKIP) {
        video.read(expected_frame);

        const auto frame = log.get_frame(frame_id).value();

        EXPECT_TRUE(images_equal(expected_frame, frame.rgb_frame));

        for (int i = 0; i < FRAME_SKIP - 1; i++) {
            video.grab();
        }
    }
}
}  // namespace robot::experimental::overhead_matching
