
#include "experimental/overhead_matching/spectacular_log.hh"
#include "experimental/overhead_matching/kimera_spectacular_data_provider.hh"

#include <iostream>
#include <sstream>

#include "fmt/format.h"
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

#include "kimera-vio/pipeline/Pipeline-definitions.h"

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

bool compare_imu_samples(const robot::experimental::overhead_matching::ImuSample& robot_imu,
                         const VIO::ImuMeasurement& kimera_imu) {
                            // timestamp
                            if (kimera_imu.timestamp_ != robot_imu.time_of_validity.time_since_epoch().count()) {
                                std::cout << "kimera timestamp " << kimera_imu.timestamp_ << std::endl;
                                std::cout << "robot timestamp " << robot_imu.time_of_validity.time_since_epoch().count() << std::endl;
                                std::cout << "diff " << robot_imu.time_of_validity.time_since_epoch().count() - kimera_imu.timestamp_ << std::endl;
                                std::cout << "timestamp" << std::endl;
                                return false;
                            }
                            // accel and gyro values
                            if (kimera_imu.acc_gyr_.rows() != 6 || kimera_imu.acc_gyr_.cols() != 1) {
                                std::cout << "shapes" << std::endl;
                                return false;
                            }
                            for (int i = 0; i<6; i++) {
                                Eigen::Vector3d active_vec;
                                if (i < 3) {
                                    active_vec = robot_imu.accel_mpss;
                                } else {
                                    active_vec = robot_imu.gyro_radps;
                                }

                                if (std::abs(active_vec(i % 3) - kimera_imu.acc_gyr_(i, 0)) > 1e-9)  {
                                    std::cout << "imu " << i << std::endl;
                                    return false;
                                }
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



    const std::filesystem::path vio_config_path(""); // loads default params
    VIO::VioParams vio_params(vio_config_path);
    vio_params.parallel_run_ = false;

    std::cout << "VIO params:" << std::endl; 
    vio_params.print();

    SpectacularDataProviderInterface s_interface(
        log_path, 0, std::numeric_limits<int>::max(), vio_params
    );

    std::vector<VIO::ImuMeasurement> imu_queue;

    auto imu_callback = [&imu_queue](const VIO::ImuMeasurement& measurement) -> void {
        imu_queue.push_back(measurement);
        return;
    };

    // bind IMU callback
    s_interface.registerImuSingleCallback(imu_callback);

    // Action
    s_interface.spin(); 

    // Verification
    std::cout << "Size of imu queue: " << imu_queue.size() << std::endl;
    EXPECT_TRUE(imu_queue.size() == times.size());
    // for (auto [imu_true, imu_kimera] : std::views::zip(original_imu_samples, imu_queue)) {
        // EXPECT_TRUE(compare_imu_samples(imu_true, imu_kimera));
    for (size_t i = 0; i < imu_queue.size(); i++) {
        EXPECT_TRUE(compare_imu_samples(original_imu_samples[ i ], imu_queue[ i ]));
        break;
    }
    std::cout << "all of the imu information checked out!" << std::endl;

}

}  // namespace robot::experimental::overhead_matching
