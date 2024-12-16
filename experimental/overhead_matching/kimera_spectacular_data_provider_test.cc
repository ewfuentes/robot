
#include "experimental/overhead_matching/kimera_spectacular_data_provider.hh"

#include <iostream>
#include <sstream>

#include "common/video/image_compare.hh"
#include "experimental/overhead_matching/spectacular_log.hh"
#include "fmt/core.h"
#include "gtest/gtest.h"
#include "kimera-vio/pipeline/Pipeline-definitions.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::overhead_matching {

std::ostream& operator<<(std::ostream& out, const time::RobotTimestamp& t) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(9)
       << std::chrono::duration<double>(t.time_since_epoch()).count();
    out << ss.str();
    return out;
}

bool compare_imu_samples(const robot::experimental::overhead_matching::ImuSample& robot_imu,
                         const VIO::ImuMeasurement& kimera_imu) {
    // timestamp
    if (kimera_imu.timestamp_ != robot_imu.time_of_validity.time_since_epoch().count()) {
        fmt::print("Kimera timestamp: {} | robot timestamp: {}", kimera_imu.timestamp_, robot_imu.time_of_validity.time_since_epoch().count());
        fmt::print("diff {}", robot_imu.time_of_validity.time_since_epoch().count() - kimera_imu.timestamp_);
        return false;
    }
    // accel and gyro values
    if (kimera_imu.acc_gyr_.rows() != 6 || kimera_imu.acc_gyr_.cols() != 1) {
        fmt::print("shapes");
        return false;
    }
    for (int i = 0; i < 6; i++) {
        Eigen::Vector3d active_vec;
        if (i < 3) {
            active_vec = robot_imu.accel_mpss;
        } else {
            active_vec = robot_imu.gyro_radps;
        }

        if (std::abs(active_vec(i % 3) - kimera_imu.acc_gyr_(i, 0)) > 1e-9) {
            fmt::print("imu ", i);
            return false;
        }
    }
    return true;
}

bool compare_bgr_frame(const std::unique_ptr<VIO::Frame>& kimera_frame, const cv::Mat& spec_bgr,
                       const time::RobotTimestamp& spec_time, const int& sequence_index) {
    // check image
    cv::Mat grey_image;
    if (spec_bgr.channels() > 1) {
        cv::cvtColor(spec_bgr, grey_image, cv::COLOR_BGR2GRAY);
    } else {
        grey_image = spec_bgr;
    }
    if (!common::video::images_equal(kimera_frame->img_, grey_image)) {
        return false;
    }
    // check timestamps
    if (kimera_frame->timestamp_ != spec_time.time_since_epoch().count()) {
        return false;
    }
    // check index number
    if (kimera_frame->id_ != sequence_index) {
        return false;
    }
    return true;
}
bool compare_depth_frame(const std::unique_ptr<VIO::DepthFrame>& kimera_depth,
                         const cv::Mat& spec_depth, const time::RobotTimestamp& spec_time,
                         const int& sequence_index) {
    // check image
    if (!common::video::images_equal(kimera_depth->depth_img_, spec_depth)) {
        return false;
    }
    // check timestamps
    if (kimera_depth->timestamp_ != spec_time.time_since_epoch().count()) {
        std::cout << "ID mismatch! " << kimera_depth->id_ << " "
                  << spec_time.time_since_epoch().count() << std::endl;
        return false;
    }
    // check index number
    if (kimera_depth->id_ != sequence_index) {
        return false;
    }
    return true;
}

TEST(KimeraSpectacularDataProviderTest, happy_case) {
    // Setup
    const std::filesystem::path log_path(
        "external/spectacular_log_snippet/recording_2024-11-21_13-36-30");
    SpectacularLog log(log_path);

    // get initial IMU information
    std::vector<double> times;
    std::vector<robot::experimental::overhead_matching::ImuSample> original_imu_samples;
    for (const double& t : log.accel_spline().ts()) {
        time::RobotTimestamp robot_time = time::RobotTimestamp() + time::as_duration(t);
        if (robot_time < log.min_imu_time() || robot_time > log.max_imu_time()) {
            continue;
        }
        times.push_back(t);
        auto maybe_sample = log.get_imu_sample(robot_time);
        EXPECT_TRUE(maybe_sample.has_value());
        original_imu_samples.push_back(maybe_sample.value());
    }

    const std::filesystem::path vio_config_path("");  // loads default params
    VIO::VioParams vio_params(vio_config_path);
    vio_params.parallel_run_ = false;

    SpectacularDataProviderInterface s_interface(log_path, 0, std::numeric_limits<int>::max(),
                                                 vio_params);

    std::vector<VIO::ImuMeasurement> imu_queue;
    std::vector<std::unique_ptr<VIO::Frame>> bgr_queue;
    std::vector<std::unique_ptr<VIO::DepthFrame>> depth_queue;

    auto imu_callback = [&imu_queue](const VIO::ImuMeasurement& measurement) -> void {
        imu_queue.push_back(measurement);
        return;
    };

    auto color_camera_callback = [&bgr_queue](std::unique_ptr<VIO::Frame> bgr_frame) -> void {
        bgr_queue.push_back(std::move(bgr_frame));
        return;
    };

    auto depth_camera_callback =
        [&depth_queue](std::unique_ptr<VIO::DepthFrame> depth_frame) -> void {
        depth_queue.push_back(std::move(depth_frame));
        return;
    };

    // bind IMU callback
    s_interface.registerImuSingleCallback(imu_callback);
    // bind images callback
    s_interface.registerLeftFrameCallback(color_camera_callback);
    // bind depth callback
    s_interface.registerDepthFrameCallback(depth_camera_callback);

    // Action
    while (s_interface.spin()) {
    }

    // Verification
    EXPECT_TRUE(imu_queue.size() == times.size());
    // for (auto [imu_true, imu_kimera] : std::views::zip(original_imu_samples, imu_queue)) {
    // EXPECT_TRUE(compare_imu_samples(imu_true, imu_kimera));
    for (size_t i = 0; i < imu_queue.size(); i++) {
        EXPECT_TRUE(compare_imu_samples(original_imu_samples[i], imu_queue[i]));
        break;
    }

    // bgr and depth images
    EXPECT_TRUE(depth_queue.size() == log.num_frames() && bgr_queue.size() == log.num_frames());
    for (size_t i = 0; i < log.num_frames(); i++) {
        std::optional<FrameGroup> fg = log.get_frame(i);
        EXPECT_TRUE(fg.has_value());

        EXPECT_TRUE(compare_bgr_frame(bgr_queue[i], fg->bgr_frame, fg->time_of_validity, i));
        EXPECT_TRUE(compare_depth_frame(depth_queue[i], fg->depth_frame, fg->time_of_validity, i));
    }
}

}  // namespace robot::experimental::overhead_matching
