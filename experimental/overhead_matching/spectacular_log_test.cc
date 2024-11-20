
#include "experimental/overhead_matching/spectacular_log.hh"

#include <iostream>
#include <sstream>

#include "gtest/gtest.h"

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
         std::cout << "t: " << sample->time_of_validity << " accel: " << sample->accel_mpss.transpose()
            << " gyro: " << sample->gyro_radps.transpose() << std::endl;
        }
    }

    std::cout << "Frame Samples (num frames: " << log.num_frames() << ")" << std::endl;
    for (int frame_id = 0; frame_id < log.num_frames(); frame_id += 100) {
        std::cout << "frame id: " << frame_id
            << " t: " << log.get_frame(frame_id)->time_of_validity << std::endl;

    }

    // Verification
    EXPECT_TRUE(false);
}
}  // namespace robot::experimental::overhead_matching
