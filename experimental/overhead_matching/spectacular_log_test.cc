
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
    SpectacularLog log("/tmp/recording/recording_2024-11-12_11-56-54");
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

    std::cout << "Frame Samples (num frames: " << log.num_frames() << ")" << std::endl;
    // cv::namedWindow("TEST DEPTH Window", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("TEST RGB Window", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("TEST Expected Window", cv::WINDOW_AUTOSIZE);
    cv::VideoCapture video("/tmp/recording/recording_2024-11-12_11-56-54/data.mov", cv::CAP_FFMPEG);
    constexpr int FRAME_SKIP = 100;
    cv::Mat expected_frame;
    for (int frame_id = 0; frame_id < log.num_frames(); frame_id += FRAME_SKIP) {
        video.read(expected_frame);

        const auto frame = log.get_frame(frame_id).value();
        std::cout << "frame id: " << frame_id << " t: " << frame.time_of_validity
                  << " depth image dims: (" << frame.depth_frame.size[0] << ", "
                  << frame.depth_frame.size[1] << ", " << frame.depth_frame.channels() << ")"
                  << " rgb image dims: (" << frame.rgb_frame.size.dims() << ", "
                  << frame.rgb_frame.size[0] << ", " << frame.rgb_frame.size[1] << ", "
                  << frame.rgb_frame.channels() << ")" << std::endl;

        // cv::imshow("TEST DEPTH Window", frame.depth_frame);
        // cv::imshow("TEST RGB Window", frame.rgb_frame);
        // cv::imshow("TEST Expected Window", expected_frame);

        EXPECT_TRUE(images_equal(expected_frame, frame.rgb_frame));

        for (int i = 0; i < FRAME_SKIP - 1; i++) {
            video.read(expected_frame);
        }
    }
}
}  // namespace robot::experimental::overhead_matching
